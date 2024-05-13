from generate_zones import DesignZones


class LanguageZones:
    def __init__(self):
        pass

    def generate_language_opt(
        self,
        shortage=None,
        balance=None,
        coverdistance=3.8,
        program_type="GE",
        balance_pct=0.15,
        shortage_pct=1.15,
    ):
        """
        For language programs,:
        - M set internally in generate_zones
        - always want contiguous
        - want to include citywide
        - must remake optimization since it uses different students and schools
        """
        self.opt = DesignZones(
            M=1,
            level="idschoolattendance",
            centroids_type=6,
            include_k8=True,
            program_type=program_type,
        )
        self.opt.balance = balance or balance_pct * float(self.opt.N) / self.opt.M
        self.opt.shortage = (
            shortage
            or -1 * shortage_pct * (sum(self.opt.seats) - self.opt.N) / self.opt.M
        )
        self.opt.construct_shortage_objective_model()
        self.opt._contiguity_const(
            max_distance=-1,
            real_distance=False,
            cover_distance=coverdistance,
            contiguity=True,
            neighbor=False,
        )

    def run_language_zone_opt(
        self, programs=None, save_path="", balance_pct=0.10, coverdistance=2.5
    ):
        """ run language program zone optimization """
        if programs is None:
            programs = ["SE", "CE", "ME", "FB", "JE", "SB", "CB"]

        not_found = programs
        while len(not_found) > 0 and balance_pct < 1:
            programs = not_found
            not_found = []
            for prog in programs:
                print("running zone opt for {} program type...".format(prog))
                self.generate_language_opt(
                    program_type=prog,
                    balance_pct=balance_pct,
                    coverdistance=coverdistance,
                )
                ans = self.opt.solve(write=False)
                if ans > 0:  # if found solution
                    self.opt.save_language_zone(save_path=save_path, prog=prog)
                else:
                    not_found.append(prog)
            # if unable to find some zone, relax constraints
            balance_pct += 0.05
            coverdistance += 0.1
            if len(not_found) > 0:
                print(
                    "ITERATION FINISHED. UNABLE TO FIND ZONES FOR {}".format(not_found)
                )
