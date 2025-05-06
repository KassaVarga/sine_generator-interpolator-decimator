# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

"(1) Генератор синуса (гармонического сигнала)"
A = np.random.randint(1, 11) # случайная амплитуда (от 1 до 10) [у.е.]
f = 0.5 # частота сигнала [Гц]
w = 2 * np.pi * f # циклическая частота сигнала [рад/с]

N = 512 # количество отсчетов сигнала [выб]
SR = 100 # частота дискретизации [выб/с]
SP = 1/SR # период дискретизации [с]
t_duration = N/SR # длительность сигнала [с]
t = np.linspace(0, t_duration, N, endpoint=False) # временной диапазон от 0 до 1 секунды [с]

signal = A * np.sin(w * t) # гармонический сигнал

"(2) Интерполятор"
K_up = 2 # коэффициент повышения частоты дискретизации сигнала
t_int = np.linspace(t[0], t[-1], int(K_up * N), endpoint=False) # временной диапазон интерполированного сигнала [с]

def linear_interpolation(signal, t, t_int): # кусочно-линейная интерполяция сигнала
    interLIN_func = interp1d(t, signal, kind='linear')
    return interLIN_func(t_int)

def sinc_interpolation(signal, N, SP, t_int): # sinc-интерполяция сигнала
    interSINC_func = np.zeros_like(t_int) # предварительное распределение расчетов
    for i, ti in enumerate(t_int):
        n = np.arange(len(signal)) # количество отсчетов сигнала на входе интерполятора [выб]
        arg = (ti - n * SP) / SP # вычисляем аргумент sinc
        interSINC_func[i] = np.sum(signal * np.sinc(arg))
    return interSINC_func

interLIN_signal = linear_interpolation(signal, t, t_int)
interSINC_signal = sinc_interpolation(signal, N, SP, t_int)

"(3) Децимация сигнала"
K_down = 2 # коэффициент понижения частоты дискретизации сигнала

def ordinary_decimation(signal, K_down): # децимация путем исключения каждого K_down'го отсчета
    return signal[::K_down]

dec_signal = ordinary_decimation(signal, K_down)
t_dec = np.linspace(t[0], t_duration, len(dec_signal), endpoint=False) # временной диапазон прореженного сигнала [с]

"(4) Интерполяция прореженного сигнала (генератор сигнала -> дециматор -> интерполятор -> выход)"
t_d_int = np.linspace(t_dec[0], t_dec[-1], K_up * len(dec_signal), endpoint=False) # временной диапазон сигнала на выходе интерполятора [с]
SR_dec = len(dec_signal)/t_dec[-1] # частота дискретизации прореженного сигнала [выб/с]
SP_dec = 1/SR_dec # # период дискретизации прореженного сигнала [с]

def lin_inter_of_dec(dec_signal, t_dec, t_d_int): # кусочно-линейная интерполяция сигнала
    interLIN_func = interp1d(t_dec, dec_signal, kind='linear') 
    return interLIN_func(t_d_int)

def sinc_inter_of_dec(dec_signal, SP_dec, t_d_int): # sinc-интерполяция сигнала
    interSINC_func = np.zeros_like(t_d_int) # предварительное распределение расчетов
    for i, ti in enumerate(t_d_int):
        n = np.arange(len(dec_signal)) # количество отсчетов сигнала на входе интерполятора [выб] 
        arg = (ti - n * SP_dec) / SP_dec # вычисляем аргумент sinc
        interSINC_func[i] = np.sum(dec_signal * np.sinc(arg))
    return interSINC_func

outputL_signal = lin_inter_of_dec(dec_signal, t_dec, t_d_int)
outputS_signal = sinc_inter_of_dec(dec_signal, SP_dec, t_d_int)

N_outL = len(outputL_signal) # количество отсчетов выходного сигнала (лин.) [выб]
N_outS = len(outputS_signal) # количество отсчетов выходного сигнала (sinc) [выб]
SR_outL = N_outL/t_d_int[-1] # частота дискретизации выходного сигнала (лин.) [выб/с]
SR_outS = N_outS/t_d_int[-1] # частота дискретизации выходного сигнала (sinc) [выб/с]

errL = ((SR_outL - SR) / SR) * 100 # ошибка между входным и выходным сигналом (лин.)
errS = ((SR_outS - SR) / SR) * 100 # ошибка между входным и выходным сигналом (sinc)

"Иллюстрация результатов"
# Генерируемый сигнал
plt.plot(t, signal, 'g-', linewidth=1.7)
plt.title('x(t) - генерируемый сигнал', fontsize=21)
plt.xlabel('t', fontsize=21, fontweight='bold')
plt.ylabel('x(t)', fontsize=21, fontweight='bold')
plt.grid()
plt.show()

# Сигнал после интерполяции (два варианта)
plt.figure(figsize=(10, 5))
plt.suptitle('x(t) - интерполированный сигнал', fontsize=21)
plt.subplot(1, 2, 1)
plt.plot(t_int, interLIN_signal, 'r-', linewidth=1.7)
plt.title('линейная', fontsize=21)
plt.xlabel('t', fontsize=21, fontweight='bold')
plt.ylabel('x(t)', fontsize=21, fontweight='bold')
plt.grid()
plt.subplot(1, 2, 2)
plt.plot(t_int, interSINC_signal, 'r-', linewidth=1.7)
plt.title('sinc', fontsize=21)
plt.xlabel('t', fontsize=21, fontweight='bold')
plt.ylabel('x(t)', fontsize=21, fontweight='bold')
plt.grid()
plt.tight_layout()
plt.show()

# Сигнал после прореживания
plt.plot(t_dec, dec_signal, 'b-', linewidth=1.7)
plt.title('x(t) - прореженный сигнал', fontsize=21)
plt.xlabel('t', fontsize=21, fontweight='bold')
plt.ylabel('x(t)', fontsize=21, fontweight='bold')
plt.grid()
plt.show()

# Интерполяция прореженного сигнала (два варианта)
plt.figure(figsize=(10, 5))
plt.suptitle('x(t) - выходной сигнал', fontsize=21)
plt.subplot(1, 2, 1)
plt.plot(t_d_int, outputL_signal, 'm-', linewidth=1.7)
plt.title('линейная', fontsize=21)
plt.xlabel('t', fontsize=21, fontweight='bold')
plt.ylabel('x(t)', fontsize=21, fontweight='bold')
plt.grid()
plt.subplot(1, 2, 2)
plt.plot(t_d_int, outputS_signal, 'm-', linewidth=1.7)
plt.title('sinc', fontsize=21)
plt.xlabel('t', fontsize=21, fontweight='bold')
plt.ylabel('x(t)', fontsize=21, fontweight='bold')
plt.grid()
plt.tight_layout()
plt.show()

"Вывод результатов"
print('=' * len('Вывод результатов'))
print('Вывод результатов')
print('=' * len('Вывод результатов'))
print()

print(f'Генерируемый сигнал: N = {len(signal)} выб, SR = {len(signal)/t_duration} выб/с')
print()

print(f'Сигнал (лин.интерполяция): N = {len(interLIN_signal)} выб, SR = {len(interLIN_signal)/t_int[-1]} выб/с')
print(f'Сигнал (sinc-интерполяция): N = {len(interSINC_signal)} выб, SR = {len(interSINC_signal)/t_int[-1]} выб/с')
print()

print(f'Сигнал (прореженный): N = {len(dec_signal)} выб, SR = {len(dec_signal)/t_dec[-1]} выб/с')
print()

print(f'Сигнал на выходе (лин.): N = {len(outputL_signal)} выб, SR = {len(outputL_signal)/t_d_int[-1]} выб/с')
print(f'Сигнал на выходе (sinc): N = {len(outputS_signal)} выб, SR = {len(outputS_signal)/t_d_int[-1]} выб/с')
print()

print(f'Ошибка: {errL} % (лин.), {errS} % (sinc)')








