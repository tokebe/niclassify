from core.StandardProgram import StandardProgram
from core.classifiers import RandomForestAC
from core.args import getargs, interactive_mode


def main():
    clf = RandomForestAC()

    program = StandardProgram(clf, getargs, interactive_mode)

    program.default_run()


if __name__ == "__main__":
    main()
