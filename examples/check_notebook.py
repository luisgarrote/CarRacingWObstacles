"""Execute the actual Colab notebook with short budgets for integration checks."""
import argparse
import os
from pathlib import Path
import nbformat
from nbclient import NotebookClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs/notebook_check.ipynb")
    args = parser.parse_args()
    # Match headless CI even when this check is run locally.
    os.environ["MPLBACKEND"] = "Agg"
    repo = Path(__file__).resolve().parents[1]
    nb = nbformat.read(repo/"examples/quickstart.ipynb", as_version=4)
    kept = [cell for cell in nb.cells if cell.get("id") not in ("install", "download")]
    kept.insert(0, nbformat.v4.new_code_cell('import os\nos.environ["SDL_VIDEODRIVER"]="dummy"\nos.environ["SDL_AUDIODRIVER"]="dummy"\nfrom pathlib import Path\nREPO = Path('+repr(str(repo))+")"))
    for cell in kept:
        if cell.get("id") == "settings":
            cell.source += "\nTRAIN_EPISODES=12; MAX_STEPS=8; ROLLOUT=8; BATCH_SIZE=4; EPOCHS=1\nVALIDATE_EVERY=4; DEVICE='cpu'\nfor case in VALIDATION_CASES: case['max_steps']=8"
        elif cell.get("id") == "video":
            cell.source = cell.source.replace('demo_case = get_case("normal_1")',
                                              'demo_case = get_case("normal_1")\ndemo_case["max_steps"] = 8')
        elif cell.get("id") == "test":
            cell.source = cell.source.replace('for case in load_cases():',
                                              'for case in load_cases():\n    case["max_steps"] = 8')
    kept.append(nbformat.v4.new_code_cell('''
import imageio.v2 as imageio
assert config["completed_episodes"] == 12
assert len(rewards) == 12
assert (RUN/"best_model.zip").is_file()
assert (RUN/"rewards.png").is_file()
assert (RUN/"best_model_evaluation.json").is_file()
assert len(results) == 10
assert {row["group"] for row in results} == {case["group"] for case in load_cases()}
selection = json.loads((RUN/"best_model.json").read_text())
assert selection["mean_return"] == max(row["mean_return"] for row in callback.summaries)
assert selection["validation_seeds"] == [case["seed"] for case in VALIDATION_CASES]
assert selection["validation_case_ids"] == [case["id"] for case in VALIDATION_CASES]
assert len(VALIDATION_CASES) == 10
assert len(callback.validation_rows) == 30
assert not {case["seed"] for case in VALIDATION_CASES}.intersection(case["seed"] for case in load_cases())
assert len(selection["groups"]) == 2
assert all(group["episodes"] == 5 for group in selection["groups"].values())
assert all(row["episodes"] == 10 for row in callback.summaries)
assert len(callback.group_summaries) == 6
assert all(row["episodes"] == 5 for row in callback.group_summaries)
assert (RUN/"validation_rewards.png").is_file()
assert len(selected_rows) == 10
assert abs(np.mean([float(row["episode_return"]) for row in selected_rows])-selection["mean_return"]) < 1e-6
reader = imageio.get_reader(str(video_path))
assert reader.count_frames() == video_result.get("simulator_frames",video_result["length"])+1
reader.close()
print("NOTEBOOK CHECK PASSED:", RUN)
'''))
    nb.cells = kept
    output = repo/args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    client = NotebookClient(nb, timeout=300, kernel_name="python3",
                            resources={"metadata":{"path":str(repo)}})
    try:
        client.execute()
        plot_cell = next(cell for cell in nb.cells if cell.get("id") == "plot")
        video_cell = next(cell for cell in nb.cells if cell.get("id") == "video")
        validation_cell = next(cell for cell in nb.cells if cell.get("id") == "validation-results")
        scenarios_cell = next(cell for cell in nb.cells if cell.get("id") == "validation-scenarios")
        assert any("image/png" in output.get("data", {}) for output in validation_cell.outputs)
        assert any("val_normal_5" in output.get("data", {}).get("text/markdown", "") for output in scenarios_cell.outputs)
        assert any("image/png" in output.get("data", {}) for output in plot_cell.outputs), "Reward graph was not embedded as a PNG in the notebook"
        assert any("<video" in output.get("data", {}).get("text/html", "") for output in video_cell.outputs)
    finally:
        nbformat.write(nb, output)
    print("Executed notebook and verified training count, best checkpoint, graph, test suite and every video frame:",output)


if __name__ == "__main__":
    main()
