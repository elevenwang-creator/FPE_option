from server.pdf_cache import PDFCache, PDFGrid
from server.greeks import Greeks
from server.interpolator import Interpolator
from server.pricer import PricingRequest, PricingResult, Pricer


struct PricingEngine:
    var cache: PDFCache
    var interpolator: Interpolator

    def __init__(out self):
        self.cache = PDFCache()
        self.interpolator = Interpolator()

    def price[
        B: Int
    ](self, requests: List[PricingRequest]) -> List[PricingResult]:
        """Unified pricing entry point. B=1 -> CPU, B>1 -> batch dispatch."""

        if len(requests) == 0:
            var empty: List[PricingResult] = []
            return empty^

        var param_hash = requests[0].param_hash
        var grid_opt = self.cache.get(param_hash)
        if not grid_opt:
            var err: List[PricingResult] = []
            err.append(
                PricingResult(
                    price=0.0, delta=0.0, gamma=0.0, vega=0.0, success=False
                )
            )
            return err^

        var pricer = Pricer[B](
            interpolator=self.interpolator.copy(),
            greeks_computer=Greeks[B](h_s=0.01, h_v=0.001),
        )
        return pricer.price(grid_opt.value(), requests)

    def store_pdf(mut self, param_hash: UInt64, var grid: PDFGrid):
        self.cache.store(param_hash, grid^)
