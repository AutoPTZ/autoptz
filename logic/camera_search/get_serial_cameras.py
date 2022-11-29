import serial.tools.list_ports


class COMPorts:
    """
    Used for finding all USB PTZ cameras
    """
    def __init__(self, data: list):
        self.data = data

    @classmethod
    def get_com_ports(cls):
        """
        Gets all USB ports being used on the computer
        :return:
        """
        data = []
        ports = list(serial.tools.list_ports.comports())

        for port_ in ports:
            obj = Object(data=dict({"device": port_.device, "description": port_.description.split("(")[0].strip()}))
            data.append(obj)

        return cls(data=data)

    @staticmethod
    def get_description_by_device(device: str):
        """
        Returns description of each device
        :param device:
        :return:
        """
        for port_ in COMPorts.get_com_ports().data:
            if port_.device == device:
                return port_.description

    @staticmethod
    def get_device_by_description(description: str):
        """
        Returns each device from their description
        :param description:
        :return:
        """
        for port_ in COMPorts.get_com_ports().data:
            if port_.description == description:
                return port_.device


class Object:
    """
    Object class to get all the device data and description needed
    """
    def __init__(self, data: dict):
        self.data = data
        self.device = data.get("device")
        self.description = data.get("description")


if __name__ == "__main__":
    """ For Local Debug and Testing """
    data_list = COMPorts.get_com_ports().data
    for port in data_list:
        if "USB" in port.description:
            print(port.device, port.description, data_list.index(port))
