w, mag, phase = signal.bode(sys)

plt.subplot(2, 1, 1)
plt.semilogx(w, mag)
plt.title('Magnitude')
plt.ylabel('Magnitude, [dB]')
plt.xlabel('$\omega[rad/s]$')

plt.subplot(2, 1, 2)
plt.semilogx(w, phase)
plt.title('Phase')
plt.ylabel('Phase, [$\circ$]')
plt.xlabel('$\omega[rad/s]$')
plt.tight_layout()