from constants.constants import Constants as c
class IdealGasData:
    """
    A class to hold equations of idea gas law calculations
    p * v = c.R * t
    """
    

    def __init__(self) -> None:
        pass

    def pv_t(self,
             p_list: list[float],
             v: float) -> list[float]:
        return [ p * v / c.R for p in p_list ]
    
    def pt_v(self,
             p_list: list[float],
             t: float) -> list[float]:
        return [ c.R * t / p for p in p_list ]
    
    def vt_p(self,
             v_list: list[float],
             t: float) -> list[float]:
        return [ c.R * t / v for v in v_list ]


class VDWGas():
    """
    A class to hold equations of van der Waals gas law calculations
    vdw gas is described by the following law:
    (p + a / v**2) * (v - b) = c.R * t
    """

    def __init__(self,
                 a: float = 0.0,
                 b: float = 0.0):
        self.a = a
        self.b = b

    def vt_p(self,
             v_list: list[float],
             t: float) -> list[float]:
        return [ c.R * t / (v - self.b) - self.a / (v * v) for v in v_list]

