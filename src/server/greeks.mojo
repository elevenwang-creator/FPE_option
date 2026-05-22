from std.algorithm import parallelize

from server.pricer import Pricer
from server.option_types import FpeParams
from server.simd_utils import vec_central_diff, vec_second_diff


struct Greeks(Copyable, Movable):
    var h_s: Float64
    var h_v: Float64

    def __init__(out self, h_s: Float64 = 0.01, h_v: Float64 = 0.001):
        self.h_s = h_s
        self.h_v = h_v

    def _bumped_params(
        self, base: FpeParams, bump_s: Float64, bump_v: Float64
    ) -> FpeParams:
        var h = base.heston.copy()
        h.S0 = h.S0 + bump_s
        h.V0 = h.V0 + bump_v
        return FpeParams(
            heston=h^,
            n_s=base.n_s,
            n_v=base.n_v,
            barrier=base.barrier,
            option_type=base.option_type,
            strikes=base.strikes.copy(),
        )

    def compute(
        self, pricer: Pricer, fpe_params: FpeParams, p_base: List[Float64]
    ) raises -> Tuple[List[Float64], List[Float64], List[Float64]]:
        var n_strikes = len(fpe_params.strikes)
        var fp_up_s = self._bumped_params(fpe_params, self.h_s, 0.0)
        var fp_dn_s = self._bumped_params(fpe_params, -self.h_s, 0.0)
        var fp_up_v = self._bumped_params(fpe_params, 0.0, self.h_v)
        var fp_dn_v = self._bumped_params(fpe_params, 0.0, -self.h_v)

        var p_up_s: List[Float64] = []
        var p_dn_s: List[Float64] = []
        var p_up_v: List[Float64] = []
        var p_dn_v: List[Float64] = []
        for _ in range(n_strikes):
            p_up_s.append(0.0)
            p_dn_s.append(0.0)
            p_up_v.append(0.0)
            p_dn_v.append(0.0)

        var up_s_ptr = p_up_s.unsafe_ptr()
        var dn_s_ptr = p_dn_s.unsafe_ptr()
        var up_v_ptr = p_up_v.unsafe_ptr()
        var dn_v_ptr = p_dn_v.unsafe_ptr()

        @parameter
        def _run_bumped(idx: Int):
            try:
                if idx == 0:
                    var r = pricer.price(fp_up_s)
                    var r_ptr = r.unsafe_ptr()
                    for k in range(n_strikes):
                        up_s_ptr[k] = r_ptr[k]
                elif idx == 1:
                    var r = pricer.price(fp_dn_s)
                    var r_ptr = r.unsafe_ptr()
                    for k in range(n_strikes):
                        dn_s_ptr[k] = r_ptr[k]
                elif idx == 2:
                    var r = pricer.price(fp_up_v)
                    var r_ptr = r.unsafe_ptr()
                    for k in range(n_strikes):
                        up_v_ptr[k] = r_ptr[k]
                else:
                    var r = pricer.price(fp_dn_v)
                    var r_ptr = r.unsafe_ptr()
                    for k in range(n_strikes):
                        dn_v_ptr[k] = r_ptr[k]
            except:
                pass

        parallelize[_run_bumped](4, 4)

        var deltas = vec_central_diff(p_up_s.copy(), p_dn_s.copy(), self.h_s)
        var gammas = vec_second_diff(
            p_up_s.copy(), p_base.copy(), p_dn_s.copy(), self.h_s
        )
        var vegas = vec_central_diff(p_up_v.copy(), p_dn_v.copy(), self.h_v)
        return Tuple(deltas^, gammas^, vegas^)
