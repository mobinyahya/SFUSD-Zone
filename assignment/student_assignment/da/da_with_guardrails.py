from itertools import zip_longest

import numpy as np

from .da import School, Student


class Student_with_class(Student):
    def __init__(self, preferences, priorities, iClass):
        super().__init__(preferences, priorities)
        self.iClass = iClass

    def __repr__(self) -> str:
        return super().__repr__() + "\n" + "Class index: " + str(self.iClass) + "\n"


class School_with_class:
    def __init__(self, index, capacity, frac_reserve_class, strictGuards):
        if round(sum(frac_reserve_class), 10) > 1:
            raise ValueError(
                "The sum of the fractions of reserved seats cannot be greater than 1"
            )

        self.school_by_classes = [
            School(index, int(capacity * frac_iClass))
            for frac_iClass in frac_reserve_class
        ]
        self.assignedPerClass = np.zeros(len(frac_reserve_class))
        has_reserves = any(fraction > 0 for fraction in frac_reserve_class)
        self.virtual_school = (
            School(-1, 0) if strictGuards and has_reserves else School(-1, capacity)
        )

    def matches(self):
        res = self.virtual_school.matches
        for school in self.school_by_classes:
            res = res.union(school.matches)

        return res

    def __repr__(self) -> str:
        return (
            "\n".join([str(school) for school in self.school_by_classes])
            + "\n THIS IS THE VIRTUAL"
            + str(self.virtual_school)
        )


class DeferredAcceptance_with_GuardRails:
    def __init__(
        self,
        school_caps,
        student_priorities,
        student_prefs,
        StudentClasses,
        frac_reserve_class,
        strictGuards,
    ):
        self.fracs, self.caps = frac_reserve_class, school_caps
        self.schools = [
            School_with_class(i, capacity, frac, strictGuards)
            for i, (capacity, frac) in enumerate(zip(school_caps, frac_reserve_class))
        ]
        self.students = [
            Student_with_class(student_pref, student_priority, idxClass)
            for (student_pref, student_priority, idxClass) in zip_longest(
                student_prefs, student_priorities, StudentClasses
            )
        ]

    def run(self):
        # import pdb
        # pdb.set_trace()
        unmatched_students = set(range(len(self.students)))

        while unmatched_students:
            for i in sorted(unmatched_students):
                student = self.students[i]
                iClass = student.iClass

                school_index = student.propose() - 1

                if student.check_exhausted(school_index):
                    unmatched_students.discard(i)
                    continue

                priority = student.priorities[int(school_index)]
                school = self.schools[int(school_index)]

                if priority < 0:
                    continue

                if school.school_by_classes[iClass].has_space():
                    student.set_match(school_index)
                    school.school_by_classes[iClass].add_match(i, priority)
                    school.virtual_school.capacity -= 1
                    unmatched_students.discard(i)

                    if school.virtual_school.has_excess_matches():
                        (
                            student_to_remove,
                            _,
                        ) = school.virtual_school.remove_lowest_priority()
                        self.students[student_to_remove].unmatch()
                        unmatched_students.add(student_to_remove)

                elif (
                    school.school_by_classes[iClass].give_lowest_priority() < priority
                    and school.school_by_classes[iClass].has_capacity()
                ):
                    student.set_match(school_index)
                    school.school_by_classes[iClass].add_match(i, priority)
                    student_to_remove, minpriority = school.school_by_classes[
                        iClass
                    ].remove_lowest_priority()

                    self.students[student_to_remove].unmatch()
                    self.students[student_to_remove].proposal_index -= 1
                    unmatched_students.add(student_to_remove)
                    unmatched_students.discard(i)

                elif school.virtual_school.has_space():
                    student.set_match(school_index)
                    school.virtual_school.add_match(i, priority)
                    unmatched_students.discard(i)

                elif (
                    school.virtual_school.give_lowest_priority() < priority
                    and school.virtual_school.has_capacity()
                ):
                    student.set_match(school_index)
                    school.virtual_school.add_match(i, priority)
                    (
                        student_to_remove,
                        _,
                    ) = school.virtual_school.remove_lowest_priority()
                    self.students[student_to_remove].unmatch()
                    unmatched_students.add(student_to_remove)
                    unmatched_students.discard(i)

        self.student_match = np.array(
            [
                student.matched_to + 1 if student.matched_to is not None else 0
                for student in self.students
            ]
        )
        self.student_proposal = np.array(
            [student.proposal_index for student in self.students]
        )
        # applied = [0, 1, 4, 5, 10, 11, 13, 15, 18, 22, 23, 24, 25, 26, 28]
        # track = [(len(self.schools[a].school_by_classes[0].matches), len(self.schools[a].school_by_classes[1].matches),  len(self.schools[a].virtual_school.matches))   for a in applied]
        # pdb.set_trace()

        return self.student_match, self.student_proposal

    def check_stability(self):
        for i in range(len(self.students)):
            iClass = self.students[i].iClass

            for k in range(self.student_proposal[i] - 1):
                school_index = self.students[i].preferences[k] - 1
                school = self.schools[school_index]

                if (
                    school.school_by_classes[iClass].give_lowest_priority()
                    < self.students[i].priorities[school_index]
                    and school.school_by_classes[iClass].has_capacity()
                ):
                    raise ValueError(
                        f"The student {i} has a higher priority ({self.students[i].priorities[school_index]}) than the school's {self.students[i].preferences[k]} lowest priority ({school.school_by_classes[iClass].give_lowest_priority()}) for the given class {iClass}"
                    )

                elif (
                    school.virtual_school.give_lowest_priority()
                    < self.students[i].priorities[school_index]
                    and school.virtual_school.has_capacity()
                ):
                    raise ValueError(
                        f"The student {i} has a higher priority ({self.students[i].priorities[school_index]}) than the virtual school's {self.students[i].preferences[k]} lowest priority ({school.virtual_school.give_lowest_priority()}) for the given class {iClass}"
                    )

        return True


class DAwithGuards:
    def __init__(
        self,
        SchoolCaps,
        StudentPrts,
        StudPrefs,
        classOfStudent,
        strictGuards=0,
        dist_student_school=None,
    ):
        self.school_capacities = SchoolCaps
        self.student_priorities = StudentPrts
        self.student_classes = classOfStudent
        self.student_prefs = StudPrefs
        self.strictGuards = strictGuards

    def setguards(self, program_reserve_frac, numOfClasses=3):
        self.program_reserve_frac = program_reserve_frac
        self.nClasses = numOfClasses

    def run(self):
        return DeferredAcceptance_with_GuardRails(
            self.school_capacities,
            self.student_priorities,
            self.student_prefs,
            self.student_classes,
            self.program_reserve_frac,
            self.strictGuards,
        ).run()
