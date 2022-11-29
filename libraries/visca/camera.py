import re
import binascii
import serial
from scipy.interpolate import interp1d


class Camera(object):
    _input = None
    _output = None
    _output_string = None
    _input_string = None

    def __init__(self, output='COM1'):
        """USB VISCA control class.

        :param output: Outbound serial port string. (default: 'COM1')
        :type output: str
        """
        self._output_string = output
        # self._input_string = input

    def init(self):
        """Initializes camera object by connecting to serial port.

        :return: Camera object.
        :rtype: Camera
        """
        # self._input = serial.Serial(self._input_string)
        self._output = serial.Serial(self._output_string)

    def command(self, com):
        """Sends hexadecimal string to serial port.

        :param com: Command string. Hexadecimal format.
        :type com: str
        :return: Success.
        :rtype: bool
        """
        try:
            self._output.write(binascii.unhexlify(com))
            return True
        except Exception as e:
            print(com, e)
            return False

    @staticmethod
    def close(serial_port):
        """Closes current serial port.

        :param serial_port: Serial port to modify.
        :return: True if successful, False if not.
        :rtype: bool
        """
        if serial_port.isOpen():
            serial_port.close()
            return True
        else:
            print("Error closing serial port: Already closed.")
            return False

    @staticmethod
    def open(serial_port):
        """Opens serial port.

        :param serial_port: Serial port to modify.
        :return: True if successful, False if not.
        :rtype: bool
        """
        if not serial_port.isOpen():
            serial_port.open()
            return True
        else:
            print("Error opening serial port: Already open.")
            return False

    def read(self, amount=3):
        total = ""
        while True:
            msg = binascii.hexlify(self._output.read())
            total = total + msg.decode()
            if msg == "ff":
                break
        return total


class D100(Camera):
    """Sony EVI-D100 VISCA control class.

    Further documentation on the VISCA protocol:
    https://pro.sony.com/bbsccms/assets/files/mkt/remotemonitoring/manuals/rm-EVID100_technical_manual.pdf
    """

    values = ["1161h", "116Dh", "122Ah", "123Ch", "12F3h", "13C2h", "151Eh", "1536h", "1844h", "226Fh", "3F2Ah",
              "40AAh", "62C9h", "82C1h"]
    y = [
        20,
        18,
        16,
        14,
        12,
        10,
        8,
        6,
        4,
        2,
        1.5,
        1,
        0.5,
        0.1]

    interp = None

    def __init__(self, output='COM1'):
        """Sony VISCA control class.

        :param output: Serial port string. (default: 'COM1')
        :type output: str
        """
        self.interp = interp1d([int(f[:-1], 16) for f in self.values], self.y)
        super(self.__class__, self).__init__(output=output)

    def init(self):
        """Initializes camera object by connecting to serial port.

        :return: Camera object.
        :rtype: Camera
        """
        super(self.__class__, self).init()
        return self

    def comm(self, com):
        """Sends hexadecimal string to serial port.

        :param com: Command string. Hexadecimal format.
        :type com: str
        :return: Success.
        :rtype: bool
        """
        super(self.__class__, self).command(com)

    def focus_near(self):
        self.command('81090448FF')
        msg = self.read(7)[4:-2]
        print(msg)
        r = ""
        if len(msg) == 8:
            for x in range(1, 9, 2):
                r += msg[x]
            x = int(r, 16)
            if x < 4449 or x > 33473:
                return None
            return self.interp(x)
        return None

    @staticmethod
    def multi_replace(text, rep):
        """Replaces multiple parts of a string using regular expressions.

        :param text: Text to be replaced.
        :type text: str
        :param rep: Dictionary of key strings that are replaced with value strings.
        :type rep: dict
        :return: Replaced string.
        :rtype: str
        """
        rep = dict((re.escape(k), v) for k, v in rep.iteritems())
        pattern = re.compile("|".join(rep.keys()))
        return pattern.sub(lambda m: rep[re.escape(m.group(0))], text)

    def relative_position(self, pan, tilt, amount_pan, amount_tilt, direction_pan=1, direction_tilt=1):
        """Moves camera relative to current position.

        :param pan: Pan speed.
        :type pan: int
        :param tilt: Tilt speed.
        :type tilt: int
        :param amount_pan: Pan amount.
        :type amount_pan: int
        :param amount_tilt: Tilt amount.
        :type amount_tilt: int
        :param direction_pan: Pan direction (1 = right, -1 = left)
        :type direction_pan: int
        :param direction_tilt: Tilt direction (1 = up, -1 = down)
        :type direction_tilt: int
        :return: True if successful, False if not.
        :rtype: bool
        """
        if direction_pan != 1:
            amount_pan = 65532 - amount_pan
        if direction_tilt != 1:
            amount_tilt = 65500 - amount_tilt
        position_string = '81010603VVWW0Y0Y0Y0Y0Z0Z0Z0ZFF'
        pan_string = "%X" % amount_pan
        pan_string = pan_string if len(pan_string) > 3 else ("0" * (4 - len(pan_string))) + pan_string
        pan_string = "0" + "0".join(pan_string)

        tilt_string = "%X" % amount_tilt
        tilt_string = tilt_string if len(tilt_string) > 3 else ("0" * (4 - len(tilt_string))) + tilt_string
        tilt_string = "0" + "0".join(tilt_string)

        rep = {"VV": str(pan) if pan > 9 else "0" + str(pan), "WW": str(tilt) if tilt > 9 else "0" + str(tilt),
               "0Y0Y0Y0Y": pan_string, "0Z0Z0Z0Z": tilt_string}

        position_string = self.multi_replace(position_string, rep)
        return self.comm(position_string)

    def home(self):
        """Moves camera to home position.

        :return: True if successful, False if not.
        :rtype: bool
        """

        return self.comm('81010604FF')

    def menu(self):
        """Opens/Closes Camera Menu

        :return: True if successful, False if not.
        :rtype: bool
        """

        return self.comm('8101060602ff')

    def zoom_in(self):
        """Zooms in Came.

        :return: True if successful, False if not.
        :rtype: bool
        """

        return self.comm('8101040702FF')

    def zoom_out(self):
        """Zooms in Came.

        :return: True if successful, False if not.
        :rtype: bool
        """

        return self.comm('8101040703FF')

    def zoom_stop(self):
        """Zooms in Came.

        :return: True if successful, False if not.
        :rtype: bool
        """

        return self.comm('8101040700FF')

    def reset(self):
        """Resets camera.

        :return: True if successful, False if not.
        :rtype: bool
        """
        return self.comm('81010605FF')

    def stop(self):
        """Stops camera movement (pan/tilt).

        :return: True if successful, False if not.
        :rtype: bool
        """
        return self.comm('8101060115150303FF')

    def cancel(self):
        """Cancels current command.

        :return: True if successful, False if not.
        :rtype: bool
        """
        return self.comm('81010001FF')

    def get_status(self, amount=5):

        self.comm('81090610FF')
        return super(self.__class__, self).read(amount=amount)

    def get_speed(self, amount=5):

        self.comm('81090611FF')
        return super(self.__class__, self).read(amount=amount)

    def _move(self, string, a1, a2):
        h1 = "%X" % a1
        h1 = '0' + h1 if len(h1) < 2 else h1

        h2 = "%X" % a2
        h2 = '0' + h2 if len(h2) < 2 else h2
        return self.comm(string.replace('VV', h1).replace('WW', h2))

    def left(self, amount=5):
        """Modifies pan speed to left.

        :param amount: Speed (0-24)
        :return: True if successful, False if not.
        :rtype: bool
        """
        hex_string = "%X" % amount
        hex_string = '0' + hex_string if len(hex_string) < 2 else hex_string
        s = '81010601VVWW0103FF'.replace('VV', hex_string).replace('WW', str(15))
        return self.comm(s)

    def right(self, amount=5):
        """Modifies pan speed to right.

        :param amount: Speed (0-24)
        :return: True if successful, False if not.
        """
        hex_string = "%X" % amount
        hex_string = '0' + hex_string if len(hex_string) < 2 else hex_string
        s = '81010601VVWW0203FF'.replace('VV', hex_string).replace('WW', str(15))
        return self.comm(s)

    def up(self, amount=5):
        """Modifies tilt speed to up.

        :param amount: Speed (0-24)
        :return: True if successful, False if not.
        """
        hs = "%X" % amount
        hs = '0' + hs if len(hs) < 2 else hs
        s = '81010601VVWW0301FF'.replace('VV', str(15)).replace('WW', hs)
        return self.comm(s)

    def down(self, amount=5):
        """Modifies tilt to down.

        :param amount: Speed (0-24)
        :return: True if successful, False if not.
        """
        hs = "%X" % amount
        hs = '0' + hs if len(hs) < 2 else hs
        s = '81010601VVWW0302FF'.replace('VV', str(15)).replace('WW', hs)
        return self.comm(s)

    def left_up(self, pan, tilt):
        return self._move('81010601VVWW0101FF', pan, tilt)

    def right_up(self, pan, tilt):
        return self._move('81010601VVWW0201FF', pan, tilt)

    def left_down(self, pan, tilt):
        return self._move('81010601VVWW0102FF', pan, tilt)

    def right_down(self, pan, tilt):
        return self._move('81010601VVWW0202FF', pan, tilt)

    def exposure_full_auto(self):
        """Changes exposure to full-auto.

        :return: True if successful, False if not.
        :rtype: bool
        """
        return self.comm('8101043900FF')

    def autofocus_sens_high(self):
        """Changes autofocus sensitivity to high.

        :return: True if successful, False if not.
        :rtype: bool
        """
        return self.comm('8101045802FF')

    def autofocus_sens_low(self):
        """Changes autofocus sensitivity to low.

        :return: True if successful, False if not.
        :rtype: bool
        """
        return self.comm('8101045803FF')

    def autofocus(self):
        """Turns autofocus on.

        :return: True if successful, False if not.
        :rtype: bool
        """
        return self.comm('8101043802FF')

    def wide_off(self):
        """Wide mode setting: Off

        Returns to original 640x480 resolution.
        :return: True if successful, False if not.
        :rtype: bool
        """
        return self.comm('8101046000FF')

    def wide_cinema(self):
        """Wide mode setting: Cinema

        Places black bars above and below picture. Otherwise maintains resolution.
        :return: True if successful, False if not.
        :rtype: bool
        """
        return self.comm('8101046001FF')

    def wide_169(self):
        """Wide mode setting: 16:9

        Stretches picture to 16:9 format.
        :return: True if successful, False if not.
        :rtype: bool
        """
        return self.comm('8101046002FF')

    def white_balance_auto(self):
        """White balance: Automatic mode

        :return: True if successful, False if not.
        :rtype: bool
        """
        return self.comm('8101043500FF')

    def white_balance_indoor(self):
        """White balance: Indoor mode

        :return: True if successful, False if not.
        :rtype: bool
        """
        return self.comm('8101043501FF')

    def white_balance_outdoor(self):
        """White balance: Outdoor mode

        :return: True if successful, False if not.
        :rtype: bool
        """
        return self.comm('8101043502FF')

    def picture_effect_off(self):
        """Off picture effect.

        :return: True if successful, False if not.
        :rtype: bool
        """
        return self.comm('8101046300FF')

    def picture_effect_pastel(self):
        """Pastel picture effect.

        :return: True if successful, False if not.
        :rtype: bool
        """
        return self.comm('8101046301FF')

    def picture_effect_negart(self):
        """Negative picture effect.

        :return: True if successful, False if not.
        :rtype: bool
        """
        return self.comm('8101046302FF')

    def picture_effect_sepia(self):
        """Sepia picture effect.

        :return: True if successful, False if not.
        :rtype: bool
        """
        return self.comm('8101046303FF')

    def picture_effect_b_w(self):
        """Black and white picture effect.

        :return: True if successful, False if not.
        :rtype: bool
        """
        return self.comm('8101046304FF')

    def picture_effect_solarize(self):
        """Solarize picture effect.

        :return: True if successful, False if not.
        :rtype: bool
        """
        return self.comm('8101046305FF')

    def picture_effect_mosaic(self):
        """Mosaic picture effect.

        :return: True if successful, False if not.
        :rtype: bool
        """
        return self.comm('8101046306FF')

    def picture_effect_slim(self):
        """Slim picture effect.

        :return: True if successful, False if not.
        :rtype: bool
        """
        return self.comm('8101046307FF')

    def picture_effect_stretch(self):
        """Stretch picture effect.

        :return: True if successful, False if not.
        :rtype: bool
        """
        return self.comm('8101046308FF')
