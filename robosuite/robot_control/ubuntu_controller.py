import socket
import numpy as np
import struct
import time


class robot_controller:
    def __init__(self):
        self.UDP_IP_IN = (
            "192.168.1.200"  # Ubuntu IP, should be the same as Matlab shows
        )
        self.UDP_PORT_IN = (
            57831  # Ubuntu receive port, should be the same as Matlab shows
        )
        self.UDP_IP_OUT = (
            "192.168.1.100"  # Target PC IP, should be the same as Matlab shows
        )
        self.UDP_PORT_OUT = 3826  # Robot 1 receive Port

        # self.s_in = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # self.s_in.bind((self.UDP_IP_IN, self.UDP_PORT_IN))
        # Receive TCP position (3*), TCP Rotation Matrix (9*), TCP Velcoity (6*), Force Torque (6*), Joint Pos
        self.unpacker = struct.Struct("12d 6d 6d 6d")

        self.robot_pose, self.robot_vel, self.TCP_wrench = None, None, None

        
    def receive(self):
        self.s_in = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.s_in.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.s_in.bind((self.UDP_IP_IN, self.UDP_PORT_IN))
        data, _ = self.s_in.recvfrom(1024)
        unpacked_data = np.array(self.unpacker.unpack(data))
        self.robot_pose, self.robot_vel, self.TCP_wrench, self.joint_pos = (
            unpacked_data[0:12],
            unpacked_data[12:18],
            unpacked_data[18:24],
            unpacked_data[24:30],
        )
        self.s_in.close()
        

    def send(self, udp_cmd):
        '''
        UDP command 1~3 TCP desired position (meters)
        UDP command 4~6 TCP desired rotation in euler ('ZYX', radians)
        UDP Kp 7~12 TCP desired stiffness Kp
        UDP Kd 13~18 TCP desired damping Kd
        UDP Mass 19_21 TCP desired mass
        UDP Interial 21~24 TCP desired inertia
        '''
        self.s_out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        udp_cmd = udp_cmd.astype("d").tostring()
        self.s_out.sendto(udp_cmd, (self.UDP_IP_OUT, self.UDP_PORT_OUT))
        self.s_out.close()

# if __name__ == "__main__": 
#     rc = robot_controller()
#     rc.receive()
#     print(rc.robot_pose)
#     udp_cmd = 10*np.ones((24,))
#     rc.send(udp_cmd)
#     print('finished')