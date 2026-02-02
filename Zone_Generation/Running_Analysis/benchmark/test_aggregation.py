"""
Test script for benchmark result aggregation and zone data export.
"""
import os
import shutil
import tempfile
import pandas as pd
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from Zone_Generation.Running_Analysis.benchmark.results import BenchmarkResult, LevelResult, aggregate_results

def test_aggregation():
    """Test the aggregation of benchmark results including per-zone export."""
    # Create a temporary directory for tests
    with tempfile.TemporaryDirectory() as tmp_dir:
        print(f"Working in temp dir: {tmp_dir}")
        
        # 1. Create dummy results
        results_dir = os.path.join(tmp_dir, "results")
        os.makedirs(results_dir)
        
        configs = [
            {"centroids_type": "4-zone", "random_seed": 42, "level": "BlockGroup_0"},
            {"centroids_type": "8-zone", "random_seed": 14, "level": "BlockGroup_0"}
        ]
        
        for i, config in enumerate(configs):
            run_dir = os.path.join(results_dir, f"run_{i}")
            os.makedirs(run_dir)
            
            # Create a simple result
            result = BenchmarkResult(
                status="OPTIMAL",
                total_wall_time=10.5 + i,
                boundary_cost=100 - i,
                config=config,
                metrics={"metric_a": 0.5, "metric_b": 0.8},
                zone_data={
                    1: {
                        "ge_students": 100, 
                        "frl_pct": 0.5, 
                        "ethnicity_pcts": {"Asian": 0.3, "White": 0.4},
                        "programs": {"GE": 10, "SA": 2}
                    },
                    2: {
                        "ge_students": 150, 
                        "frl_pct": 0.4,
                        "ethnicity_pcts": {"Asian": 0.2, "White": 0.5},
                        "programs": {"GE": 12}
                    }
                }
            )
            
            # Save it
            result.save(run_dir)
        
        print("Created dummy results.")
        
        # 2. Run aggregation
        summary_csv = os.path.join(tmp_dir, "summary.csv")
        zone_data_dir = os.path.join(tmp_dir, "export")
        
        print("Running aggregation...")
        df = aggregate_results(
            results_dir, 
            output_file=summary_csv,
            recompute_metrics=False,  # Don't need real graph for this test
            zone_data_folder=zone_data_dir
        )
        
        # 3. Verify results
        print("\nVerifying...")
        
        # Check summary CSV
        assert os.path.exists(summary_csv), "Summary CSV not created"
        df_loaded = pd.read_csv(summary_csv)
        assert len(df_loaded) == 2, f"Expected 2 results, got {len(df_loaded)}"
        assert "metric_a" in df_loaded.columns
        assert "config_centroids_type" in df_loaded.columns
        print("✓ Summary CSV verified")
        
        # Check zone data export
        assert os.path.exists(zone_data_dir), "Zone data folder not created"
        exported_files = os.listdir(zone_data_dir)
        assert len(exported_files) == 2, f"Expected 2 exported CSVs, got {len(exported_files)}"
        
        # Check content of one exported file
        sample_csv = os.path.join(zone_data_dir, exported_files[0])
        df_zone = pd.read_csv(sample_csv)
        assert len(df_zone) == 2, "Expected 2 zones in exported file"
        assert "ge_students" in df_zone.columns
        assert "eth_Asian" in df_zone.columns, "Ethnicity flattening failed"
        assert "prog_GE" in df_zone.columns, "Programs flattening failed"
        print("✓ Per-zone data export verified")
        
        print("\n✓ SUCCESS: Aggregation test passed!")

if __name__ == "__main__":
    try:
        test_aggregation()
    except Exception as e:
        print(f"\n✗ FAILURE: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
