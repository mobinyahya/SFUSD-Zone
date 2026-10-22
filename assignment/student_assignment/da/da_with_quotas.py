"""Created 8/19/20.

@author Itai Ashlagi

deferred acceptance algorithm with guards
"""

import numpy as np


class DaWithCapSplit:
    def __init__(
        self,
        SchoolCaps,
        StudentPrts,
        StudPrefs,
        seats_by_zone,
        student_zones,
        strictGuards=0,
    ):
        self.SchoolCaps = SchoolCaps
        self.StudentPrts = StudentPrts
        self.StudPrefs = StudPrefs

        # print(StudentPrts.shape)
        self.n = StudentPrts.shape[0]  # number of students
        self.s = StudentPrts.shape[1]  # number of programs
        self.L = StudPrefs.shape[1]  # Max preference length

        self.strictGuards = strictGuards

        self.seats_by_zone = seats_by_zone

        # print(student_zones)
        for i in range(len(student_zones)):
            if student_zones[i] > -1:  # To skip nan values
                student_zones[i] = int(student_zones[i])
            else:
                student_zones[i] = -1
        self.student_zones = student_zones
        self.numOfClasses = self.seats_by_zone.shape[0]

        # self.schoolsToReserve =  np.zeros([self.s])
        # for i in

    def run(self):
        StudentProposal = np.zeros(
            [self.n]
        )  # Index of school on student's list that they will propose to next
        # StudentRankBasedOnRealPref = np.zeros([n]) #rank accroding to real prefs

        self.StudentMatched = np.zeros(
            [self.n]
        )  # Indicator vector for which students are currently matched

        self.LowestPriority = np.zeros(
            [self.s, self.numOfClasses, 2]
        )  # LowestPriority[i,j, 0] is index of student with lowest priority assigned to school i,
        # LowestPriority[i,j, 1] is the student's priority at school i

        self.SchoolMatches = []  # List of sets of students assigned to each school
        for k in range(0, self.s):
            self.SchoolMatches.append(set())

        self.assignedPerClass = np.zeros([self.s, self.numOfClasses])

        StillUmatchedStudents = {self.n - 1}
        for i in range(1, self.n):
            StillUmatchedStudents.add(self.n - i - 1)

        # print(self.StudentPrts)

        while bool(StillUmatchedStudents):
            i = StillUmatchedStudents.pop()
            if self.student_zones[i] < 0:
                self.StudentMatched[i] = -1
            # print('run',i)
            while self.StudentMatched[i] == 0:
                # print(self.student_zones[i])
                iClass = int(self.student_zones[i])
                index = int(StudentProposal[i])
                StudentProposal[i] += 1
                # print('rfrfr',i)
                if StudentProposal[i] == self.L:
                    self.StudentMatched[i] = -1
                    continue
                school = int(self.StudPrefs[i, index]) - 1
                if not self.isFeasible(i, school, self.StudentPrts):
                    continue
                r = True
                for p in range(0, 9):
                    if self.seats_by_zone[p, school] > 0:
                        r = False
                        break
                if (
                    self.seats_by_zone[int(self.student_zones[i]), school] == 0
                    and not r
                ):
                    continue

                if school >= 0:  # if a real school
                    priority = self.StudentPrts[i, school]
                    reserveSchool = False
                    if self.seats_by_zone[int(self.student_zones[i]), school] == 0:
                        # if self.schoolsToReserve[school] == 0:  #if the school is not a program we want reserve
                        iClass = 0

                    else:
                        iClass = int(self.student_zones[i])
                        reserveSchool = True
                    if (
                        len(self.SchoolMatches[school]) < self.SchoolCaps[school]
                    ):  # if the school has seats
                        # if i == 0:
                        #   print('aaa',i, priority, school)
                        # add the student in this case depending on the stirct rules or not
                        # print("School:",isinstance(school,int),school)
                        # print("iClass:",isinstance(iClass,int),iClass)
                        if (
                            self.LowestPriority[school, iClass, 1] == 0
                        ):  # no one was assigned yet to this class
                            # if i == 0:
                            #   print('aaaaaaaa',i, priority, school, iClass)
                            self.LowestPriority[school, iClass, 0] = i
                            self.LowestPriority[school, iClass, 1] = priority

                        if (
                            self.strictGuards and reserveSchool
                        ):  # self.schoolsToReserve[school] == 1:
                            if (
                                self.assignedPerClass[school, iClass]
                                >= self.seats_by_zone[iClass, school]
                            ):
                                (
                                    rejectedstudent,
                                    classtoReject,
                                ) = self._leastPriorityStudent(
                                    i, iClass, school, priority
                                )
                                if i != rejectedstudent and self.SchoolCaps[school] > 0:
                                    # print(self.SchoolMatches[school])
                                    # print('a',i,iClass, rejectedstudent, classtoReject)
                                    # print('b',self.seats_by_zone[iClass,school])
                                    # print('c',self.seats_by_zone[classtoReject,school])
                                    self.SchoolMatches[school].remove(rejectedstudent)
                                    self.StudentMatched[rejectedstudent] = 0
                                    if (
                                        StudentProposal[rejectedstudent] == self.L
                                        or self.StudPrefs[
                                            rejectedstudent,
                                            int(StudentProposal[rejectedstudent]),
                                        ]
                                        == 0
                                    ):
                                        self.StudentMatched[rejectedstudent] = -1
                                    # print(StudPrefs)
                                    else:
                                        StillUmatchedStudents.add(rejectedstudent)
                                    self.SchoolMatches[school].add(i)
                                    self.StudentMatched[i] = 1

                                    if classtoReject != iClass:
                                        self.assignedPerClass[school, iClass] += 1
                                        self.assignedPerClass[
                                            school, classtoReject
                                        ] -= 1

                                    # Updates the lowest priority student
                                    self._UpdatePriorities(
                                        i,
                                        iClass,
                                        priority,
                                        rejectedstudent,
                                        classtoReject,
                                        school,
                                    )
                            else:
                                # if i == 0:
                                #   print('aaa',i,school)
                                self.SchoolMatches[school].add(i)
                                self.StudentMatched[i] = 1
                                self.assignedPerClass[school, iClass] += 1
                                self._UpdatePriorities(
                                    i, iClass, priority, i, iClass, school
                                )
                        else:
                            # if i == 0:
                            #   print('xxx',i,school, iClass)
                            self.SchoolMatches[school].add(i)
                            self.StudentMatched[i] = 1
                            self.assignedPerClass[school, iClass] += 1
                            self._UpdatePriorities(
                                i, iClass, priority, i, iClass, school
                            )

                    else:  # if we need to reject someone (no more seats)
                        # print(i, iClass, school, priority)
                        rejectedstudent, classtoReject = self._leastPriorityStudent(
                            i, iClass, school, priority
                        )
                        # print('kk',i, rejectedstudent,classtoReject)
                        if i != rejectedstudent and self.SchoolCaps[school] > 0:
                            self.SchoolMatches[school].remove(rejectedstudent)
                            self.StudentMatched[rejectedstudent] = 0
                            if (
                                StudentProposal[rejectedstudent] == self.L
                                or self.StudPrefs[
                                    rejectedstudent,
                                    int(StudentProposal[rejectedstudent]),
                                ]
                                == 0
                            ):
                                self.StudentMatched[rejectedstudent] = -1
                            else:
                                StillUmatchedStudents.add(rejectedstudent)
                            # if i == 0:
                            #   print('qqq',i,school)
                            self.SchoolMatches[school].add(i)
                            self.StudentMatched[i] = 1

                            if classtoReject != iClass:
                                self.assignedPerClass[school, iClass] += 1
                                self.assignedPerClass[school, classtoReject] -= 1

                            # Updates the lowest priority student
                            self._UpdatePriorities(
                                i,
                                iClass,
                                priority,
                                rejectedstudent,
                                classtoReject,
                                school,
                            )
                            # self.LowestPriority[school,0] = i
                            # self.LowestPriority[school,1] = priority
                            # for teststudent in self.SchoolMatches[school]:
                            #    if self.StudentPrts[teststudent,school] < self.LowestPriority[school,1]:
                            #        self.LowestPriority[school,0] = teststudent
                            #        self.LowestPriority[school,1] = self.StudentPrts[teststudent,school]

                z = int(StudentProposal[i])
                if self.StudentMatched[i] == 0:
                    if StudentProposal[i] == self.L or self.StudPrefs[i, z] == 0:
                        self.StudentMatched[i] = -1

        self.StudentMatch = np.zeros([self.n])  # Student's match is 0 if unmatched
        for j in range(0, self.s):
            for student in self.SchoolMatches[j]:
                self.StudentMatch[student] = j + 1

        # return StudentMatch, SchoolMatches
        return self.StudentMatch, StudentProposal

    def _leastPriorityStudent(self, i, iClass, school, priority):
        # return the rejected student

        reserveSchool = self.seats_by_zone[int(self.student_zones[i]), school] > 0
        # print('GGGGGGGGGGG',reserveSchool, i, int(self.student_zones[i]), iClass, self.seats_by_zone[int(self.student_zones[i]),school])
        # print('llla', i, iClass, school, priority)
        if not reserveSchool:  # self.schoolsToReserve[school] == 0:
            if priority > self.LowestPriority[school, 0, 1]:
                #       print('lllb', i, iClass, school, priority)
                #       print('xxx', self.seats_by_zone[iClass,school], self.assignedPerClass[school,iClass])
                # for j in range(0,9):
                # print(j, self.seats_by_zone[j,school],  self.assignedPerClass[school,j])
                return int(self.LowestPriority[school, 0, 0]), iClass
            # if i == 0:
            #   print('rrrr',i,iClass,school, priority)
            #  print('lllc', i, iClass, school, priority)
            return i, iClass
        else:
            classToReject = self._classToReject(school, i, iClass, reserveSchool)
            if iClass != classToReject:
                # print('xx',int(self.LowestPriority[school, classToReject, 0]), classToReject, school, self.LowestPriority[school, classToReject, 1])
                # print(self.SchoolCaps[school], self.assignedPerClass[school,0])
                # print('llld', i, iClass, classToReject, school, priority)
                return int(self.LowestPriority[school, classToReject, 0]), classToReject
            if priority < self.LowestPriority[school, classToReject, 0]:
                # print(priority, i, self.LowestPriority[school, classToReject, 0], self.LowestPriority[school, classToReject, 1])
                # print('bb',i,classToReject)
                # if i == 0:
                #   print('ssss',i,iClass,school, priority)
                return i, classToReject
            # print('cc',int(self.LowestPriority[school, classToReject, 0]), classToReject)
            # if int(self.LowestPriority[school, classToReject, 0])==0:
            #       print('ttt',i,iClass,school, priority)
            #       print('ppp',int(self.LowestPriority[school, classToReject, 0]))
            # print('llle', i, iClass, classToReject, school, priority)
            return int(self.LowestPriority[school, classToReject, 0]), classToReject

    def _classToReject(self, school, i, iClass, reserveSchool=False):
        rejectClass = 0
        if not reserveSchool:  # self.schoolsToReserve[school] == 0:
            # print('fffff')
            return 0
        tmp = -10000
        # self.assignedPerClass[school,0]-self.seats_by_zone[0,school]
        # print('dddd',i,iClass,school,tmp,self.seats_by_zone[0,school])
        rejectClass = i
        for j in range(0, self.numOfClasses):
            if (
                self.seats_by_zone[j, school] > 0
                and self.assignedPerClass[school, j] > 0
            ):
                tmpX = self.assignedPerClass[school, j] - self.seats_by_zone[j, school]
                if tmpX > tmp:
                    tmp = tmpX
                    rejectClass = j
                    # print('hhh',j,self.seats_by_zone[j,school], self.assignedPerClass[school,j], tmp)

        # print('RejectClass', rejectClass)
        return rejectClass

    def _UpdatePriorities(
        self, i, iClass, priority, rejectedStudent, classToReject, school
    ):
        reserveSchool = self.seats_by_zone[int(self.student_zones[i]), school] > 0
        if not reserveSchool:
            iClass = 0
            classToReject = 0
        if iClass == classToReject:
            self.LowestPriority[school, iClass, 0] = i
            self.LowestPriority[school, iClass, 1] = priority
            for teststudent in self.SchoolMatches[school]:
                tmpClass = int(self.student_zones[teststudent])
                if not reserveSchool:
                    tmpClass = 0
                if tmpClass == iClass:
                    if (
                        self.StudentPrts[teststudent, school]
                        < self.LowestPriority[school, iClass, 1]
                    ):
                        self.LowestPriority[school, iClass, 0] = teststudent
                        self.LowestPriority[school, iClass, 1] = self.StudentPrts[
                            teststudent, school
                        ]
            return

        if not reserveSchool:
            iClass = 0
        if (
            priority < self.LowestPriority[school, iClass, 1]
            or self.LowestPriority[school, iClass, 1] == 0
        ):
            self.LowestPriority[school, iClass, 0] = i
            self.LowestPriority[school, iClass, 1] = priority

        tmp = 1000000
        for teststudent in self.SchoolMatches[school]:
            tmpClass = int(self.student_zones[teststudent])
            if not reserveSchool:
                tmpClass = 0
            if tmpClass == classToReject:
                if (
                    self.StudentPrts[teststudent, school] < tmp
                    and teststudent != rejectedStudent
                ):
                    self.LowestPriority[school, classToReject, 0] = teststudent
                    self.LowestPriority[school, classToReject, 1] = self.StudentPrts[
                        teststudent, school
                    ]
                    tmp = self.LowestPriority[school, classToReject, 1]

    def isFeasible(self, student, school, StudentPrts):
        # if StudentPrts[student,school] < -50:
        #   print('infeasbile')
        return StudentPrts[student, school] > -50
