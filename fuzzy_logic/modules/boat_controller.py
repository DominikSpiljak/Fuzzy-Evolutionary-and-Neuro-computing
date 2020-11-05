import sys
from accel_fuzzy_system import AccelFuzzySystemMin
from steer_fuzzy_system import SteerFuzzySystemMin
from defuzziers import COADefuzzifier


def main():

    defuzzier = COADefuzzifier()
    fsAkcel = AccelFuzzySystemMin(defuzzier)
    fsKormilo = SteerFuzzySystemMin(defuzzier)

    while True:
        inpstr = input()

        if inpstr == 'KRAJ':
            return

        (L, D, LK, DK, V, S) = [int(s)
                                for s in inpstr.split(" ") if s.isdigit()]

        akcel = fsAkcel.conclude(L, D, LK, DK, V, S)
        kormilo = fsKormilo.conclude(L, D, LK, DK, V, S)

        print(akcel, kormilo)
        sys.stdout.flush()


if __name__ == "__main__":
    main()
