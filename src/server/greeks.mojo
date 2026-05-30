from server.pricer import Pricer
from server.option_types import FpeParams
from server.simd_utils import vec_central_diff, vec_second_diff


struct Greeks(Copyable, Movable):
    var h_s: Float64
    var h_v: Float64
    var num_insert: Int

    def __init__(
        out self,
        h_s: Float64 = 0.01,
        h_v: Float64 = 0.001,
        num_insert: Int = 50,
    ):
        self.h_s = h_s
        self.h_v = h_v
        self.num_insert = num_insert

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
        self, fpe_params: FpeParams, p_base: List[Float64]
    ) raises -> Tuple[List[Float64], List[Float64], List[Float64]]:
        var pricer = Pricer(num_insert=self.num_insert)
        var fp_up_s = self._bumped_params(fpe_params, self.h_s, 0.0)
        var fp_dn_s = self._bumped_params(fpe_params, -self.h_s, 0.0)
        var fp_up_v = self._bumped_params(fpe_params, 0.0, self.h_v)
        var fp_dn_v = self._bumped_params(fpe_params, 0.0, -self.h_v)

        var p_up_s = pricer.price(fp_up_s)
        var p_dn_s = pricer.price(fp_dn_s)
        var p_up_v = pricer.price(fp_up_v)
        var p_dn_v = pricer.price(fp_dn_v)

        var deltas = vec_central_diff(p_up_s, p_dn_s, self.h_s)
        var gammas = vec_second_diff(p_up_s, p_base.copy(), p_dn_s, self.h_s)
        var vegas = vec_central_diff(p_up_v, p_dn_v, self.h_v)
        return Tuple(deltas^, gammas^, vegas^)
