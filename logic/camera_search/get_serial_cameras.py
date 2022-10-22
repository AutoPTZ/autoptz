import serial.tools.list_ports

class COMPorts:

    def __init__(self, data: list):
        self.data = data

    @classmethod
    def get_com_ports(cls):
        data = []
        ports = list(serial.tools.list_ports.comports())

        for port_ in ports:
            obj = Object(data=dict({"device": port_.device, "description": port_.description.split("(")[0].strip()}))
            data.append(obj)

        return cls(data=data)

    @staticmethod
    def get_description_by_device(device: str):
        for port_ in COMPorts.get_com_ports().data:
            if port_.device == device:
                return port_.description

    @staticmethod
    def get_device_by_description(description: str):
        for port_ in COMPorts.get_com_ports().data:
            if port_.description == description:
                return port_.device


class Object:
    def __init__(self, data: dict):
        self.data = data
        self.device = data.get("device")
        self.description = data.get("description")


if __name__ == "__main__":
    for port in COMPorts.get_com_ports().data:
        print(port.device, port.description)
