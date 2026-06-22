# ICON4Py / gt4py-dace — exner-AoS codegen transform (AMD MI300A).
#
# Productionizes the validated "exner-AoS" optimization for the icon4py thetarho program
# `compute_rho_theta_pgrad_and_update_vn`. The dominant terrain kernels gather 3 cell fields
# (d2dz2 / ddz / temporal_extrapolation of the perturbed exner function) via a doubly-indirect
# `field[E2C[edge], k+ikoffset]` access — an uncached scatter (~10.6% L2, ~12.5% line use), so
# the 3 SEPARATE arrays cost 3 scattered 64-byte line-fetches per site. We pack them into one
# interleaved `aos[N][3]` (a coalesced repack kernel fills it) so ONE fetch serves all 3, and
# redirect ONLY the terrain (ikoffset) reads — the flat `temporal` reads keep the original array
# (no coalescing penalty). A coalesced repack kernel rebuilds the AoS each call from the 3 fields.
# Measured on MI300A: -15.5% on thetarho, correct for the real model. See
# amd_scripts/docs/EXNER_AOS_OPTIMIZATION.md.
#
# This is a POST-CODEGEN transform (operates on the generated .cu/.cpp in the DaCe build folder)
# because gt4py lays out a [Cell,K,3] frontend field's pack dim STRIDED, not unit-stride, so the
# AoS benefit cannot be expressed at the frontend; it needs direct codegen/layout control. It is
# self-checking and fails closed (returns False -> no patch, original correct code) on any codegen
# mismatch. Measured -14.3% (1.266 -> 1.085 ms) on real MI300A, correctness-passing (N=3562240).
#
# REFUTED (do not re-add): the "upstream" variant -- having the producer
# `compute_perturbed_quantities_and_interpolation` pre-write the packed AoS so thetarho skips the
# repack -- looked like -20% on thetarho IN ISOLATION but is NET-NEGATIVE at the model level: the
# AoS is stride-3, so the producer's interleaved writes are uncoalesced and cost +23.6% (+~100us)
# on the producer, which EXCEEDS the ~57us thetarho saves. The dedicated in-kernel repack does the
# strided write more efficiently than smearing it across the producer.
#
# OPT-IN via GT4PY_EXNER_AOS=1 (thetarho only). N (= Cell*K) is HARDCODED to the regional benchmark
# grid (_N_REGIONAL=3562240), the production target on beverin. A DIFFERENT grid needs _N_REGIONAL
# adjusted -- auto-deriving N from the SDFG was tried and removed (array-name ambiguity returned a
# tiny wrong N -> OOB crash). That hardcoding is why it is opt-in, not silent default-on.
import os
import pathlib
import re

PROGRAM = "compute_rho_theta_pgrad_and_update_vn"
_D2 = "d2dz2_of_temporal_extrapolation_of_perturbed_exner_on_model_levels"
_DZ = "ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels"
_TE = "temporal_extrapolation_of_perturbed_exner"
_N_REGIONAL = 3562240  # Cell * K for icon_benchmark_regional (the production grid)

_REPACK_CU = (
    "__global__ void __repack_exner_aos_kernel(const double * __restrict__ d2, const double * __restrict__ dz, const double * __restrict__ te, double * __restrict__ aos, int Nexn) {\n"
    "    int i = (blockIdx.x * blockDim.x) + threadIdx.x;\n"
    "    if (i < Nexn) { aos[(i*3)+0] = d2[i]; aos[(i*3)+1] = dz[i]; aos[(i*3)+2] = te[i]; }\n"
    "}\n"
    "void __dace_runkernel_repack_exner_aos(" + PROGRAM + "_state_t *__state, const double * __restrict__ d2, const double * __restrict__ dz, const double * __restrict__ te, double * __restrict__ aos, int Nexn) {\n"
    "    void *args[] = { (void*)&d2, (void*)&dz, (void*)&te, (void*)&aos, (void*)&Nexn };\n"
    "    gpuError_t __err = hipLaunchKernel((void*)__repack_exner_aos_kernel, dim3((Nexn+255)/256, 1, 1), dim3(256, 1, 1), args, 0, nullptr);\n"
    "    DACE_KERNEL_LAUNCH_CHECK(__err, \"__repack_exner_aos_kernel\", (Nexn+255)/256, 1, 1, 256, 1, 1);\n"
    "}\n")
_FWDDECL = "void __dace_runkernel_repack_exner_aos(" + PROGRAM + "_state_t *, const double *, const double *, const double *, double *, int);\n"

_READ_RE = re.compile(r"__tlet_field\[\(\((__\w+_K_stride) \* __tlet_index_K\) \+ __tlet_index_Cell\)\]")


def _scale(m):
    return "__tlet_field[(3 * ((%s * __tlet_index_K) + __tlet_index_Cell))]" % m.group(1)


def apply_exner_aos(build_dir, n=_N_REGIONAL):
    """Apply the exner-AoS transform to the generated source in a DaCe build folder.

    n = Cell*K (the exner trio element count) for this grid; pass derive_n(sdfg). The repack
    kernel rebuilds the AoS from the 3 fields on every call (correct for changing inputs).

    Returns True iff all expected edits matched (18 terrain reads, 1 repack kernel, and the
    host-side struct/malloc/free/fwd-decl/repack-call/3 terrain-call redirects). Idempotent.
    """
    n = str(int(n))
    build = pathlib.Path(build_dir)
    cu = list(build.rglob("*_cuda.cpp")) + list(build.rglob("*.cu"))
    cpp = [f for f in build.rglob("*.cpp")
           if "_cuda" not in f.name and "SampleMain" not in f.name and "CMake" not in str(f)]
    tot = dict(reads=0, kernel=0, struct=0, malloc=0, free=0, fwddecl=0, fill=0, calls=0)

    for f in cu:
        t = f.read_text()
        if "__repack_exner_aos_kernel" in t:
            return True  # already patched
        t, _nsub = _READ_RE.subn(_scale, t)
        if _nsub:
            i = t.index("__global__ void  __launch_bounds__")
            t = t[:i] + _REPACK_CU + "\n" + t[i:]
            tot["kernel"] += 1
            f.write_text(t)
            tot["reads"] += _nsub

    for f in cpp:
        t = f.read_text()
        if "aos_exner" in t:
            continue
        if "gpu_context;" not in t and "__dace_runkernel" not in t:
            continue
        t2 = t.replace("dace::cuda::Context *gpu_context;",
                       "dace::cuda::Context *gpu_context;\n    double * __restrict__ aos_exner;\n    bool aos_filled;", 1)
        tot["struct"] += (t2 != t); t = t2
        anchor = "DACE_GPU_CHECK(hipMalloc((void**)&__state->__0_gtir_tmp_83,"
        if anchor in t:
            line = t[t.index(anchor):].split("\n", 1)[0]
            t = t.replace(line, line + "\n    DACE_GPU_CHECK(hipMalloc((void**)&__state->aos_exner, 3 * %s * sizeof(double)));\n    __state->aos_filled = false;" % n, 1)
            tot["malloc"] += 1
        fa = "DACE_GPU_CHECK(hipFree(__state->__0_gtir_tmp_83));"
        if fa in t:
            t = t.replace(fa, fa + "\n    DACE_GPU_CHECK(hipFree(__state->aos_exner));", 1); tot["free"] += 1
        m = re.search(r"^DACE_EXPORTED void __dace_runkernel_map_", t, re.M)
        if m:
            t = t[:m.start()] + _FWDDECL + t[m.start():]; tot["fwddecl"] += 1
        m = re.search(r"\n(        __dace_runkernel_map_)", t)
        if m:
            call = "        __dace_runkernel_repack_exner_aos(__state, %s, %s, %s, __state->aos_exner, %s);\n" % (_D2, _DZ, _TE, n)
            t = t[:m.start()] + "\n" + call + t[m.start() + 1:]; tot["fill"] += 1
        out = []
        for ln in t.split("\n"):
            is_call = ln.lstrip().startswith("__dace_runkernel_map_") and "(__state," in ln
            if is_call and (" %s," % _D2) in ln:
                ln = ln.replace(" %s," % _D2, " __state->aos_exner,").replace(
                    " %s," % _DZ, " __state->aos_exner + 1,").replace(
                    " %s," % _TE, " __state->aos_exner + 2,")
                tot["calls"] += 1
            out.append(ln)
        f.write_text("\n".join(out))

    ok = (tot["reads"] == 18 and tot["kernel"] == 1 and tot["struct"] == 1 and tot["malloc"] == 1
          and tot["free"] == 1 and tot["fwddecl"] == 1 and tot["fill"] == 1 and tot["calls"] == 3)
    print("[exner_aos] %s: %s N=%s %s" % (PROGRAM, "PATCHED" if ok else "MISMATCH", n, tot), flush=True)
    return ok
