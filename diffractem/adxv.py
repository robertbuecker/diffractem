"""
Adxv remote control.

Inspired by:
https://github.com/erikssod/adxv_class by Daniel Eriksson (MIT license)
https://github.com/keitaroyam/yamtbx by Keitaro Yamashita (BSD license)

"""
import socket
import subprocess
import time
import logging

class Adxv:

    def __init__(self, adxv_bin=None, hdf5_path='/entry/data/raw_counts', **kwargs):

        self.logger = logging.getLogger()
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(fmt=('[%(levelname)s] %(name)s ''%(funcName)s | %(message)s')))
        self.logger.handlers = [handler]
        self.logger.setLevel('INFO') # or INFO, or DEBUG, etc

        self.logger = logging.getLogger(__name__)

        self.adxv_bin = adxv_bin
        self.adxv_opts = kwargs

        if self.adxv_bin is None:
            self.adxv_bin = "adxv"

        self.hdf5_path = hdf5_path
        self.adxv_proc = None  # subprocess object
        self.adxv_port = 8100  # adxv's default port. overridden later.
        self.sock = None

        self.spot_type_counter = -1

    def start(self, cwd=None):

        if not self.is_alive():

            # find available port number
            self.logger.debug('Searching for available port number')
            sock_test = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock_test.bind(("localhost", 0))
            self.adxv_port = sock_test.getsockname()[1]
            sock_test.close()
            self.logger.debug(f'Port {self.adxv_port} will be used for adxv. Attempting to connect.')

            # build adxv start command
            adxv_comm = self.adxv_bin + ' -socket {} -hdf5dataset {}'.format(self.adxv_port, self.hdf5_path)

            for opt, val in self.adxv_opts.items():
                adxv_comm += ' -{} {}'.format(opt, val)

            # start adxv
            self.logger.debug(f'adxv command is: \n {adxv_comm}')
            self.adxv_proc = subprocess.Popen(adxv_comm, shell=True, cwd=cwd)

            for i in range(10):  # try for 5 seconds.
                try:
                    self.sock = socket.socket(socket.AF_INET,
                                              socket.SOCK_STREAM)  # On OSX(?), need to re-create object when failed
                    self.sock.connect(("localhost", self.adxv_port))
                    self.logger.info('Connected to Port {}'.format(self.adxv_port))
                    break
                except socket.error as err:
                    self.logger.debug('Waiting for socket connection...')
                    time.sleep(.5)
                    continue

    def is_alive(self):
        return self.adxv_proc is not None and self.adxv_proc.poll() is None  # None means still running.

    def send(self, payload):
        '''
        Takes command, encodes it, and sends it down the socket.
        '''

        self.start()

        try:
            self.logger.debug("payload = {}".format(payload))
            self.sock.sendall(payload.encode())

        except Exception as e:
            self.logger.error(e)

    def load_image(self, image_file: str):
        '''
        Load an image file
        '''
        payload = 'load_image %s\n' % (image_file)
        self.send(payload)

    def raise_window(self, window: str):
        '''
        Raises a Window. <window> must be one of
        'Control', 'Image', 'Magnify', 'Line', or
        'Load'.
        '''
        payload = 'raise_window %s\n' % (window)
        self.send(payload)

    def raise_image(self):
        '''
        Raises image window; see raise_window for
        additional options but this seems like the
        most common one.
        '''
        payload = 'raise_window Image\n'
        self.send(payload)

    def save_image(self, path_name_format: str):
        '''
        Save an image file (jpeg or tiff)
        '''
        payload = 'save_image %s\n' % (path_name_format)
        self.send(payload)

    def slab(self, N: int):
        '''
        Display slab N
        '''
        payload = 'slab %i\n' % (N)
        self.send(payload)

    def set_slab(self, N: int):
        '''
        Same as slab, but don’t load the image
        '''
        payload = 'set_slab %i\n' % (N)
        self.send(payload)

    def slabs(self, N: int):
        '''
        Slab thickness to display
        '''
        payload = 'slabs %i\n' % (N)
        self.send(payload)

    def set_slabs(self, N: int):
        '''
        Same as slabs, but don’t load the image
        '''
        payload = 'set_slabs %i\n' % (N)
        self.send(payload)

    def exit(self):
        '''
        Exit Adxv
        '''
        payload = 'exit\n'
        self.send(payload)

    def stride(self, N: int):
        """
        stride - sets Stride in the Load Window
        """
        payload = 'stride %i\n' % (N)
        self.send(payload)

    def increment_slabs(self):
        """
        increment_slabs - checks the +Slabs checkbox in the Load Window
        """
        payload = 'increment_slabs\n'
        self.send(payload)

    def increment_files(self):
        """
        increment_files - unchecks the +Slabs checkbox in the Load Window
        """
        payload = 'increment_files\n'
        self.send(payload)

    def contrast_min(self, N: int):
        """
        contrast_min - sets the min contrast value
        """
        payload = 'contrast_min %i\n' % (N)
        self.send(payload)

    def contrast_max(self, N: int):
        """
        contrast_max - sets the max contrast value
        """
        payload = 'contrast_max %i\n' % (N)
        self.send(payload)

    def define_spot(self, color, radius=0, box=0, group=None):

        if group is None:
            self.spot_type_counter += 1
        else:
            self.spot_type_counter = group

        self.send('box %d %d\n' % (box, box))  # seems ignored?
        self.send('define_type %d color %s radius %d\n' % (group, color, radius))

        return self.spot_type_counter

    def load_spots(self, spots):
        #if len(spots) == 0:
        #    return

        self.send("load_spots %d\n" % len(spots))

        for x, y, t in spots:
            self.send("%.2f %.2f %d\n" % (x, y, t))

        self.send("end_of_pack\n")
