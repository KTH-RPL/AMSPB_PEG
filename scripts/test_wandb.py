# test_wandb.py
import wandb
import random
import json
import os

def main():
    # Initialize a run in your project (change project name if you like)
    run = wandb.init(
        project="test-artifact-upload",
        job_type="upload-test",
        reinit=True,   # allow multiple runs in the same process
    )

    # Create some dummy data
    sample = {"value": random.random()}

    # Write it out to a file
    filename = "sample.json"
    with open(filename, "w") as f:
        json.dump(sample, f)

    # Create an artifact, add the file, and log it
    artifact = wandb.Artifact(
        name="random-sample",
        type="dataset",
        description="A single-sample JSON with a random float"
    )
    artifact.add_file(filename)
    run.log_artifact(artifact, aliases=["latest", "v1"]).wait()

    # Optional: confirm where it went
    print(f"Uploaded artifact {artifact.name}:{artifact.version}")

    # Clean up local file
    os.remove(filename)

    # Finish the run
    run.finish()


if __name__ == "__main__":
    main()
