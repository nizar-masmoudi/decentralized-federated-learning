from client.models import ConvNet
import math

P = 20  # dBm => mW
W = 20e6  # 20 MHz
N0 = -174  # dBm/Hz => mW/Hz
Obj = 1e6  # 1 Gb
D = 100


def convert(value: float):  # dBm -> mW
    return 10 ** ((value - 30) / 10)


def channel_gain(dist: float) -> float:
    return -38 - 20 * math.log10(dist)


def transmission_rate(power: float, bandwidth: float, dist: float):
    return bandwidth * math.log2(1 + (convert(power) * convert(channel_gain(dist))) / (convert(N0) * bandwidth))


r = transmission_rate(P, W, 500)
t = Obj / r
e = t * convert(P)

print('{:<30} | {:<15} | {:<15} |'.format('Transmission power', f'{P} dBm', f'{convert(P)} mW'))
print('{:<30} | {:<15} | {:<15} |'.format('Bandwidth', f'{W/1e6:.0f} MHz', f'{W:.0f} Hz'))
print('{:<30} | {:<15} | {:<15} |'.format('Noise power spectral density', f'{N0} dBm/Hz', f'{convert(N0):.3e} mW/Hz'))
print('{:<30} | {:<15} | {:^15} |'.format('Distance', f'{D} m', ''))
print('{:<30} | {:<15} | {:<15} |'.format('Object size', f'{Obj/1e9:.0f} Gb', f'{Obj:.0f} bits'))
print('-'*68)
print('{:<30} | {:<15} | {:^15} |'.format('Channel gain', f'{channel_gain(D)} dB', ''))
print('{:<30} | {:<15} | {:^15} |'.format('Transmission rate', f'{r/1e6:.3f} Mb/s', f'{r:.3f} bits/s'))
print('-' * 10 + ' Communication energy ' + '-' * 10)
print('Channel gain = {:.2f} dB'.format(channel_gain(100)))
print('Transmission rate = {:.2f} Mb/s'.format(r / 1e6))
print('Transmission time = {:.5f} s'.format(t))
print('Energy consumption = {:.5f} J'.format(e))
#
local_iters = 5
kappa = 1e-28
flops = ConvNet().flops
ds_len = 5000
cpu_frequency = 1e9
fpc = 4

e = local_iters*kappa*flops*ds_len*(cpu_frequency**2)/fpc

print()
print('-' * 10 + ' Computation energy ' + '-' * 10)
print('Energy consumption = {:.5f} J'.format(e))
