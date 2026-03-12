class ValveControl:
    def __init__(self):
        self.is_open = False

    def open(self):
        self.is_open = True

    def close(self):
        self.is_open = False


class ValveCheck:
    def __init__(self):
        self.is_in = True


class Pump:
    def __init__(self):
        self.speed = 0

    def set_speed(self, speed):
        self.speed = speed


class Filter:
    def __init__(self):
        self.clean = True


class Motor:
    def __init__(self):
        self.rpm = 50

    def set_speed(self, rpm):
        if rpm >= 0 and rpm < 500:
            self.rpm = rpm


# Sensors
class TemperatureSensor:
    def __init__(self):
        self.value = 0

    def read(self):
        return self.value

class PressureSensor:
    def __init__(self):
        self.value = 0

    def read(self):
        return self.value

class OxygenSensor:
    def __init__(self):
        self.value = 0

    def read(self):
        return self.value

class OpticalDensitySensor:
    def __init__(self):
        self.value = 0

    def read(self):
        return self.value

class FluidLevelSensor:
    def __init__(self):
        self.value = 0

    def read(self):
        return self.value

class FoamSensor:
    def __init__(self):
        self.value = 0

    def read(self):
        return self.value

# COMPONENTS

class AirLine:
    def __init__(self):
        self.check_valve = ValveCheck()
        self.control_valve = ValveControl()
        self.filter = Filter()

class GasExit:
    def __init__(self):
        self.control_valve = ValveControl()

class AntiFoamLine:
    def __init__(self):
        self.pump = Pump()
        self.check_valve = ValveCheck()
        self.filter = Filter()

class BaseLine:
    def __init__(self):
        self.pump = Pump()
        self.check_valve = ValveCheck()
        self.filter = Filter()

class AcidLine:
    def __init__(self):
        self.pump = Pump()
        self.check_valve = ValveCheck()
        self.filter = Filter()

class SubstrateLine:
    def __init__(self):
        self.pump = Pump()
        self.check_valve = ValveCheck()
        self.filter = Filter()

#TANK
class Tank:
    max_level = 3 #liters
    def __init__(self):
        self.temperature = TemperatureSensor()
        self.pressure = PressureSensor()
        self.oxygen = OxygenSensor()
        self.od = OpticalDensitySensor()
        self.level = FluidLevelSensor()
        self.foam = FoamSensor()
        self.motor = Motor()

# Sensory System
class SensorySystem:

    def __init__(self, tank):
        self.tank = tank

    def read_all_sensors(self):
        data = {
            "temperature": self.tank.temperature.read(),
            "pressure": self.tank.pressure.read(),
            "oxygen": self.tank.oxygen.read(),
            "optical_density": self.tank.od.read(),
            "fluid_level": self.tank.level.read(),
            "foam_level": self.tank.foam.read()
        }
        return data
