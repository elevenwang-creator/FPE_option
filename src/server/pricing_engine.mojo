from server.pricer import Pricer, PDFGrid
from server.greeks import Greeks
from server.option_types import FpeParams, PricingResult


struct PricingEngine:
    var rtol: Float64
    var atol: Float64
    var num_insert: Int

    def __init__(out self, rtol: Float64 = 1e-4, atol: Float64 = 1e-6, num_insert: Int = 50):
        self.rtol = rtol
        self.atol = atol
        self.num_insert = num_insert

    def price(self, fpe_params: FpeParams) raises -> List[PricingResult]:
        if not fpe_params.is_valid():
            var err: List[PricingResult] = []
            err.append(
                PricingResult(
                    price=0.0, delta=0.0, gamma=0.0, vega=0.0, success=False
                )
            )
            return err^

        var pricer = Pricer(
            rtol=self.rtol, atol=self.atol, num_insert=self.num_insert
        )
        var p_base = pricer.price(fpe_params)

        var greeks = Greeks()
        var g = greeks.compute(pricer, fpe_params, p_base)
        var deltas = g[0].copy()
        var gammas = g[1].copy()
        var vegas = g[2].copy()

        var results: List[PricingResult] = []
        for k in range(len(p_base)):
            results.append(
                PricingResult(
                    price=p_base[k],
                    delta=deltas[k],
                    gamma=gammas[k],
                    vega=vegas[k],
                    success=True,
                )
            )
        return results^
