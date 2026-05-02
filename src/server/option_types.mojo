# Centralised option-pricing types for the FPE engine.

from engines.nais.fbsde import FBSDEParams
from engines.nais.nais_net import NaisNet


@fieldwise_init
struct OptionParams(Copyable, Movable):
    var S: Float64
    var K: Float64
    var V: Float64
    var barrier: Float64
    var option_type: Int

    def is_valid(self) -> Bool:
        return (
            self.S > 0.0
            and self.K > 0.0
            and self.V >= 0.0
            and (self.barrier == 0.0 or self.barrier > self.S)
            and self.option_type >= 0
            and self.option_type <= 3
        )


@fieldwise_init
struct PricingResult(Copyable, Movable, Writable):
    var price: Float64
    var delta: Float64
    var gamma: Float64
    var vega: Float64
    var success: Bool


@fieldwise_init
struct RoughBergomiParams(Copyable, Movable, Hashable):
    var H: Float64
    var eta: Float64
    var rho: Float64
    var r: Float64
    var T: Float64
    var S0: Float64
    var V0: Float64
    var epsilon_t: Float64
    var M: Int
    var N: Int
    var D: Int

    def to_fbsde_params(self) -> FBSDEParams:
        return FBSDEParams(
            Xi=[self.S0, self.V0],
            T=self.T,
            M=self.M,
            N=self.N,
            D=self.D,
            H=self.H,
            eta=self.eta,
            pho=self.rho,
            r=self.r,
            epsilon_t=self.epsilon_t,
        )


@fieldwise_init
struct NAISModel(Copyable, Movable):
    var net: NaisNet
    var params: RoughBergomiParams
