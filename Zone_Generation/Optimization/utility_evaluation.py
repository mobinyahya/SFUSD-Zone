import pandas as pd
import numpy as np

from scipy.special import logsumexp

class UtilityEvaluator:
    def __init__(self, utility_path, student_path):
        """
        utility_path: Path to the MNL model utility matrix CSV.
        student_path: Path to the student data CSV mapping students to blocks.
        """
        self.utility_path = utility_path
        self.student_path = student_path
        self.utility_df = None
        self.student_df = None
        self.school_to_cols = {}

    def load_data(self):
        """Loads and pre-processes utility and student data."""
        print(f"Loading utility matrix from {self.utility_path}...")
        self.utility_df = pd.read_csv(self.utility_path)
        # Strip '2324-' from studentno and convert to int for easier matching
        # The prefix might change, so we take the last part after '-'
        self.utility_df['studentno'] = self.utility_df['studentno'].apply(
            lambda x: int(x.split('-')[-1]) if isinstance(x, str) and '-' in x else x
        )
        
        print(f"Loading student data from {self.student_path}...")
        self.student_df = pd.read_csv(self.student_path)
        
        # Map school IDs to columns in utility matrix
        # Columns are like '413-GE-KG'
        cols = self.utility_df.columns[1:] # first is studentno
        self.school_to_cols = {}
        for col in cols:
            school_id = col.split('-')[0]
            if school_id not in self.school_to_cols:
                self.school_to_cols[school_id] = []
            self.school_to_cols[school_id].append(col)
        
        # Ensure studentno is numeric in student_df as well
        self.student_df['studentno'] = pd.to_numeric(self.student_df['studentno'], errors='coerce')

    def _get_merged_data(self, zone_dict, G, level):
        """Internal helper to prepare student-to-zone mappings."""
        if self.utility_df is None or self.student_df is None:
            self.load_data()

        if level is None:
            level = G.graph.get('level', 'BlockGroup')
            if isinstance(level, str) and '_' in level:
                level = level.split('_')[0]

        # 1. Map each zone to its set of schools
        zone_to_schools = {}
        for node_id, zone_id in zone_dict.items():
            if zone_id not in zone_to_schools:
                zone_to_schools[zone_id] = set()
            
            node_key = node_id
            if node_key not in G.nodes:
                try: node_key = int(node_id)
                except (ValueError, TypeError): pass
            
            if node_key in G.nodes:
                school_ids = G.nodes[node_key].get('school_ids', [])
                zone_to_schools[zone_id].update([str(sid) for sid in school_ids])

        # 2. Map blocks/blockgroups to zones
        area_to_zone = {}
        for node_id, zone_id in zone_dict.items():
            node_key = node_id
            if node_key not in G.nodes:
                try: node_key = int(node_id)
                except (ValueError, TypeError): pass
            
            if node_key in G.nodes:
                if 'block_ids' in G.nodes[node_key]:
                    for bid in G.nodes[node_key]['block_ids']:
                        area_to_zone[str(int(float(bid)))] = zone_id
                elif 'area_id' in G.nodes[node_key]:
                    area_id = G.nodes[node_key]['area_id']
                    area_to_zone[str(int(float(area_id)))] = zone_id
        
        # 3. Map students to zones
        student_col = 'census_blockgroup' if level == 'BlockGroup' else 'census_block'
        merged_df = self.student_df.merge(self.utility_df, on='studentno')
        
        def get_zone(val):
            if pd.isnull(val): return None
            return area_to_zone.get(str(int(float(val))))

        merged_df['assigned_zone'] = merged_df[student_col].apply(get_zone)
        initial_count = len(merged_df)
        merged_df = merged_df.dropna(subset=['assigned_zone'])
        if len(merged_df) < initial_count:
            print(f"Warning: {initial_count - len(merged_df)} students were not assigned to any zone.")

        # 4. Map zones to utility columns
        zone_to_cols = {}
        for zone_id, schools in zone_to_schools.items():
            cols = []
            for sid in schools:
                cols.extend(self.school_to_cols.get(sid, []))
            zone_to_cols[zone_id] = [c for c in cols if c in merged_df.columns]

        return merged_df, zone_to_cols, student_col

    def evaluate(self, zone_dict, G, level=None, method='max'):
        """
        Calculates utility metrics based on the zoning.
        
        method: 'max' (previous default) or 'logsum' (standard MNL welfare).
        """
        merged_df, zone_to_cols, student_col = self._get_merged_data(zone_dict, G, level)
        
        print(f"Calculating utility metrics using method='{method}'...")
        results = []
        for zone_id, group in merged_df.groupby('assigned_zone'):
            cols = zone_to_cols.get(zone_id, [])
            if not cols:
                group['utility'] = -np.inf
            else:
                if method == 'max':
                    group['utility'] = group[cols].max(axis=1)
                elif method == 'logsum':
                    # logsumexp for numerical stability
                    group['utility'] = logsumexp(group[cols].values, axis=1)
            results.append(group)
        
        if results:
            evaluated_df = pd.concat(results)
        else:
            evaluated_df = merged_df.copy()
            evaluated_df['utility'] = np.nan
        
        # Aggregate by block
        block_utilities = evaluated_df.groupby(student_col)['utility'].sum()

        return {
            'student_utilities': evaluated_df[['studentno', 'utility', 'assigned_zone']],
            'block_utilities': block_utilities
        }


    def get_utility_impact_gradients(self, zone_dict, G, level=None, method='max'):
        r"""
        Calculates the change in utility for every student and block if a school was added or removed.
        
        If school s is in zone z, 'remove' impact is U(z) - U(z \ {s}).
        If school s is NOT in zone z, 'add' impact is U(z \cup {s}) - U(z).
        
        Returns a dictionary:
        {
            'student_impacts': DataFrame with columns [studentno, census_block, school_id, type, impact],
            'block_impacts': DataFrame with columns [census_block, school_id, type, impact]
        }
        """
        merged_df, zone_to_cols, student_col = self._get_merged_data(zone_dict, G, level)
        
        # 1. Map each school ID to its CURRENT zone
        school_to_current_zone = {}
        for node_id, zone_id in zone_dict.items():
            node_key = node_id
            if node_key not in G.nodes:
                try: node_key = int(node_id)
                except: pass
            if node_key in G.nodes:
                for sid in G.nodes[node_key].get('school_ids', []):
                    school_to_current_zone[str(sid)] = zone_id

        all_schools = sorted(self.school_to_cols.keys())

        print(f"Calculating granular utility impacts (add/remove) using method='{method}'...")
        student_impact_records = []
        
        for zone_id, group in merged_df.groupby('assigned_zone'):
            cols = zone_to_cols.get(zone_id, [])
            
            # Precalculate baseline per student in this zone
            if method == 'logsum':
                if not cols:
                    baseline_utils = np.full(len(group), -1e10)
                else:
                    baseline_utils = logsumexp(group[cols].values, axis=1)
            else:
                if not cols:
                    baseline_utils = np.full(len(group), -1e10)
                else:
                    baseline_utils = group[cols].max(axis=1).values
            
            for sid in all_schools:
                sid_cols = [c for c in self.school_to_cols[sid] if c in group.columns]
                if not sid_cols: continue
                
                is_in_zone = (school_to_current_zone.get(sid) == zone_id)
                sid_utils_matrix = group[sid_cols].values
                
                if method == 'logsum':
                    sid_logsum = logsumexp(sid_utils_matrix, axis=1)
                    diff = sid_logsum - baseline_utils
                    
                    if is_in_zone:
                        # Removal impact: U_new - U = ln(1 - exp(V_s - U))
                        safe_diff = np.minimum(diff, -1e-15)
                        student_impacts = np.log1p(-np.exp(safe_diff))
                        type_str = 'remove'
                    else:
                        # Addition impact: U_new - U = ln(1 + exp(V_s - U))
                        student_impacts = np.log1p(np.exp(diff))
                        type_str = 'add'
                else: # max
                    sid_max = sid_utils_matrix.max(axis=1)
                    if is_in_zone:
                        remaining_cols = [c for c in cols if c not in sid_cols]
                        if not remaining_cols:
                            new_utils = np.full(len(group), -1e10)
                        else:
                            new_utils = group[remaining_cols].max(axis=1).values
                        student_impacts = new_utils - baseline_utils
                        type_str = 'remove'
                    else:
                        new_utils = np.maximum(baseline_utils, sid_max)
                        student_impacts = new_utils - baseline_utils
                        type_str = 'add'

                # Create records for each student in the group
                # To be memory efficient, we only record non-zero impacts if possible, 
                # but the requirement says "for every block and student"
                temp_df = pd.DataFrame({
                    'studentno': group['studentno'].values,
                    student_col: group[student_col].values,
                    'school_id': sid,
                    'type': type_str,
                    'impact': student_impacts
                })
                student_impact_records.append(temp_df)
        
        if not student_impact_records:
            return {
                'student_impacts': pd.DataFrame(columns=['studentno', student_col, 'school_id', 'type', 'impact']),
                'block_impacts': pd.DataFrame(columns=[student_col, 'school_id', 'type', 'impact'])
            }

        student_impact_df = pd.concat(student_impact_records, ignore_index=True)
        
        # Aggregate by block
        block_impact_df = student_impact_df.groupby([student_col, 'school_id', 'type'])['impact'].sum().reset_index()
        
        return {
            'student_impacts': student_impact_df,
            'block_impacts': block_impact_df
        }

if __name__ == "__main__":
    # Example usage (can be adjusted for testing)
    utility_path = "/share/data/school_choice/simulation-files/choice-model/estimates_2324_exp8_0514.csv"
    student_path = "/share/data/school_choice/Data/Cleaned/r1_filter_student_without_specialprogs_2324.csv"
    
    # This is just a placeholder for testing the structure
    # To run a real test, you'd need a zone_dict and graph G
    print("UtilityEvaluator class defined. Ready for use.")
