import numpy as np


# recursively find a cycle in the graph and return the string
def detectCycle(studentFavs, schoolFavs, current, cycle):
    int_i = int(float(current[1:]))

    # build the string for the node we'll search next
    next_current = ("S" if current[0] == "s" else "s") + str(
        studentFavs[int_i] if current[0] == "s" else schoolFavs[int_i]
    )

    try:
        # we've found a cycle, return cycle string from that point forwards
        index = cycle.index(next_current)
        return cycle[index:] + [current]
    except ValueError:
        pass

    # keep searching until we enter a cycle
    return detectCycle(studentFavs, schoolFavs, next_current, cycle + [current])


def TTC(capacities, priorities, pref):
    nStudents = len(pref)
    nSchools = len(capacities)

    studentMatch = [-1] * nStudents
    studentRank = [0] * nStudents

    studentFavs = [-1] * nStudents
    schoolFavs = [-1] * nSchools

    unassigned = nStudents
    firstUnassigned = 0

    # init pointing sets
    pointingAtStudents = []
    pointingAtSchools = []

    for i in range(nStudents):
        pointingAtStudents.append(set())
    for i in range(nSchools):
        pointingAtSchools.append(set())

    lastSchoolIndex = [0] * nSchools

    priorities_t = np.transpose(priorities)
    priority_order = np.argsort(-priorities_t)

    pref = pref - np.ones(pref.shape)
    # repoint to the highest-ranked available school
    def reassignStudent(student):
        nonlocal unassigned
        nonlocal firstUnassigned

        student = int(student)

        pointingAtSchools[int(pref[student, int(studentRank[student])])].remove(
            student
        )

        while True:
            studentRank[student] += 1

            if (
                studentRank[student] >= len(pref[student])
                or pref[student, studentRank[student]] == -1
            ):
                # if the student would rather go unmatched or is out of schools,
                # save as -2 which removes them from the market
                studentMatch[student] = -2
                studentFavs[student] = -2
                unassigned -= 1
                if student == firstUnassigned:
                    while studentMatch[firstUnassigned] != -1:
                        firstUnassigned += 1
                return

            preferred = int(pref[student, studentRank[student]])

            if capacities[preferred] == 0:
                continue

            pointingAtSchools[preferred].add(student)
            studentFavs[student] = preferred
            return

    # repoint to the highest-ranked available student
    def reassignSchool(school):
        school = int(school)
        pointingAtStudents[
            int(priority_order[school, lastSchoolIndex[school]])
        ].remove(school)

        while True:
            lastSchoolIndex[school] += 1
            try:
                preferred = int(priority_order[school, lastSchoolIndex[school]])
            except IndexError:
                # we're out of students in this iteration, keep moving
                return

            if studentMatch[preferred] != -1:
                continue

            pointingAtStudents[preferred].add(school)
            schoolFavs[school] = preferred
            return

    # save the initial highest choice for each student
    for i in range(nStudents):
        preferred = int(pref[i, 0])
        studentFavs[i] = preferred
        pointingAtSchools[preferred].add(i)
    # save the initial highest available choice for each school
    for i in range(nSchools):
        # make sure input is well-formed, no empty schools to start with
        # assert capacities[i] > 0, "school initialized with 0 capacity"
        preferred = int(priority_order[i, 0])
        schoolFavs[i] = preferred
        pointingAtStudents[preferred].add(i)

    # breakpoint()

    # while there are unassigned students remaining:
    while unassigned > 0:
        # get the first unassigned student and search for a cycle
        initial = str(firstUnassigned)
        cycle = detectCycle(studentFavs, schoolFavs, "s" + initial, [])

        # breakpoint()

        # get all students in cycle, assign them
        for item in [x for x in cycle if x[0] == "s"]:
            studentIndex = int(float(item[1:]))
            studentMatch[studentIndex] = studentFavs[studentIndex]
            studentFavs[studentIndex] = -1
            unassigned -= 1
            if studentIndex == firstUnassigned:
                while (
                    firstUnassigned < len(studentMatch)
                    and studentMatch[firstUnassigned] != -1
                ):
                    firstUnassigned += 1

            try:
                pointingAtSchools[studentMatch[studentIndex]].remove(
                    studentIndex
                )
            except KeyError:
                pass

            if unassigned == 0:
                break

            # repoint any schools pointing at the student we just assigned
            items = [x for x in pointingAtStudents[studentIndex]]
            for i in items:
                reassignSchool(i)

            if studentMatch[studentIndex] != -2:
                capacities[studentMatch[studentIndex]] -= 1
                # if school is now empty, repoint students pointing at it
                if capacities[studentMatch[studentIndex]] == 0:
                    items = [
                        x for x in pointingAtSchools[studentMatch[studentIndex]]
                    ]
                    for i in items:
                        reassignStudent(i)

            if unassigned == 0:
                break

    for i in range(0, nStudents):
        studentRank[i] += 1
        studentMatch[i] += 1

        # Checks if student is matched to a school they have very negative priority to
        if studentMatch[i] > 0:
            if priorities[i, int(studentMatch[i] - 1)] < -50:
                studentMatch[i] = 0
                studentRank[i] = 99

    return [studentMatch, studentRank]
