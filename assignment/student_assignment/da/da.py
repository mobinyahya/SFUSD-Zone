import heapq
from collections import OrderedDict
from itertools import zip_longest

import numpy as np


class School:
    def __init__(self, index, capacity):
        self.index = index
        self.capacity = capacity
        self.matches = set()
        self.priority_storage = OrderedDict()
        self._priority_heap = []
        self.lowest_priority_match = None

    def is_full(self):
        return len(self.matches) >= self.capacity

    def has_space(self):
        return self.capacity > len(self.matches) and self.capacity > 0

    def has_capacity(self):
        return self.capacity > 0

    def has_excess_matches(self):
        return bool(self.matches) and self.capacity < len(self.matches)

    def give_lowest_priority(self):
        if self.lowest_priority_match is None:
            return -np.inf

        if self._current_min_priority() != self.lowest_priority_match[1]:
            raise ValueError(
                "Priority storage and lowest priority match do not match"
            )

        return self.lowest_priority_match[1]

    def give_full_lowest_priority(self):
        if self.lowest_priority_match is None:
            return -np.inf, None

        if self._current_min_priority() != self.lowest_priority_match[1]:
            raise ValueError(
                "Priority storage and lowest priority match do not match"
            )

        return self.lowest_priority_match[1], self.lowest_priority_match[0]

    def _current_min_priority(self):
        while (
            self._priority_heap
            and self._priority_heap[0] not in self.priority_storage
        ):
            heapq.heappop(self._priority_heap)
        return self._priority_heap[0]

    def add_match(self, student, priority):
        self.matches.add(student)

        if priority in self.priority_storage:
            self.priority_storage[priority].append(student)
        else:
            self.priority_storage[priority] = [student]
            heapq.heappush(self._priority_heap, priority)

        if (
            self.lowest_priority_match is None
            or priority < self.lowest_priority_match[1]
        ):
            self.lowest_priority_match = (student, priority)

    def remove_lowest_priority(self):
        min_priority = self._current_min_priority()
        min_student = self.priority_storage[min_priority].pop(0)

        self.matches.remove(min_student)

        if len(self.priority_storage[min_priority]) == 0:
            del self.priority_storage[min_priority]

        if len(self.priority_storage) == 0:
            self.lowest_priority_match = None

        else:
            new_min_prior = self._current_min_priority()
            new_min_student = self.priority_storage[new_min_prior][0]

            self.lowest_priority_match = (new_min_student, new_min_prior)

        return min_student, min_priority

    def __repr__(self) -> str:
        return (
            "School index: "
            + str(self.index)
            + "\n"
            + "Capacity: "
            + str(self.capacity)
            + "\n"
            + "Matches: "
            + str(self.matches)
            + "\n"
            + "Priority storage: "
            + str(self.priority_storage)
            + "\n"
            + "Lowest priority match: "
            + str(self.lowest_priority_match)
            + "\n"
        )


class Student:
    def __init__(self, preferences, priorities):
        self.preferences = preferences
        self.priorities = priorities
        self.matched, self.rejects = False, False
        self.matched_to = None
        self.proposal_index = 0

    def propose(self):
        if self.proposal_index >= len(self.preferences):
            return -1

        self.proposal_index += 1
        return self.preferences[self.proposal_index - 1]

    def check_exhausted(self, school_index):
        if school_index == -2:
            self.matched, self.rejects = True, True
            self.matched_to = None
            return True
        return False

    def unmatch(self):
        self.matched, self.matched_to = False, None

    def set_match(self, school):
        self.matched, self.matched_to = True, school

    def __repr__(self) -> str:
        return (
            "Student preferences: "
            + str(self.preferences)
            + "\n"
            + "Student priorities: "
            + str(self.priorities)
            + "\n"
            + "Matched: "
            + str(self.matched)
            + "\n"
            + "Matched to: "
            + str(self.matched_to)
            + "\n"
            + "Rejects: "
            + str(self.rejects)
            + "\n"
            + "Proposal index: "
            + str(self.proposal_index)
            + "\n"
        )


class DeferredAcceptance:
    def __init__(
        self,
        school_caps,
        student_priorities,
        student_prefs,
        idx2studentno=None,
        studentno2idx=None,
        program_indicies=None,
    ):
        self.schools = [
            School(i, capacity) for i, capacity in enumerate(school_caps)
        ]
        self.students = [
            Student(student_pref, student_priority)
            for (student_pref, student_priority) in zip_longest(
                student_prefs, student_priorities
            )
        ]
        self.idx2studentno = idx2studentno
        self.studentno2idx = studentno2idx
        self.program2idx = program_indicies
        self.idx2program = (
            {value: key for key, value in program_indicies.items()}
            if program_indicies
            else None
        )

    def run(self):
        unmatched_students = set(range(len(self.students)))
        while unmatched_students:
            for i in sorted(unmatched_students):
                student = self.students[i]
                school_index = student.propose() - 1
                school = self.schools[int(school_index)]
                priority = student.priorities[int(school_index)]

                if student.check_exhausted(school_index):
                    unmatched_students.discard(i)
                    continue

                if priority < 0:
                    continue

                if school.is_full():
                    school_lowest_priority = school.give_lowest_priority()

                    if (
                        priority < school_lowest_priority
                        or not school.has_capacity()
                    ):
                        continue

                    else:
                        student_to_remove, _ = school.remove_lowest_priority()
                        self.students[student_to_remove].unmatch()
                        unmatched_students.add(student_to_remove)

                        student.set_match(school_index)
                        school.add_match(
                            i, student.priorities[int(school_index)]
                        )
                        unmatched_students.discard(i)

                else:
                    student.set_match(school_index)
                    school.add_match(i, student.priorities[int(school_index)])
                    unmatched_students.discard(i)

        self.student_match = np.array(
            [
                student.matched_to + 1 if student.matched_to is not None else 0
                for student in self.students
            ]
        )
        self.lowest_priority = np.array(
            [
                school.lowest_priority_match[1]
                if school.lowest_priority_match is not None
                else 0
                for school in self.schools
            ]
        )
        self.student_proposal = np.array(
            [student.proposal_index for student in self.students]
        )

        return self.student_match, self.lowest_priority, self.student_proposal

    def check_stability(self):
        for i in range(len(self.students)):
            for k in range(self.student_proposal[i] - 1):
                school_index = int(self.students[i].preferences[k]) - 1
                school = self.schools[school_index]

                if (
                    school.give_lowest_priority()
                    < self.students[i].priorities[school_index]
                ):
                    break
        return True
