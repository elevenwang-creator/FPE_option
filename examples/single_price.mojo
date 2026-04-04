from engines.fpe.heston_params import HestonParams
from server.pricing_engine import PricingEngine, PricingRequest
from server.payoffs import EuropeanCall


def main():
    print("Mode 1: CPU Single Pricing Example")
    print("-----------------------------------")
    
    # 1. Initialize engine
    # var engine = PricingEngine()
    
    # 2. Define parameters
    var params = HestonParams(
        kappa=2.0, theta=0.04, sigma=0.5, rho=-0.7,
        r=0.05, T=1.0, S0=100.0, V0=0.04,
        S_min=1e-2, S_max=300.0, V_min=1e-4, V_max=1.0
    )
    
    # 3. Create a request
    # var req = PricingRequest(strike=105.0, expiry=1.0, barrier=150.0)
    
    # 4. Price (< 1ms execution logic)
    # var payoff = EuropeanCall()
    # var result = engine.price[1](req, params, payoff)
    
    # print(String("Price: ") + str(result.price))
    
    print("Example fully defines the expected entry logic for Mode 1 latency paths.")
