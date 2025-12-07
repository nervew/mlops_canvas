"""Build and push the FastAPI image to Azure Container Registry.

This script assumes Azure CLI and Docker are installed and that the user has
permissions over the target subscription.
"""

import argparse
import pathlib
import subprocess


def run(command: list[str]) -> None:
    print(f"[cmd] {' '.join(command)}")
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and push FastAPI image to ACR")
    parser.add_argument("--subscription", default="Gobierno de datos", help="Azure subscription name or ID")
    parser.add_argument("--resource-group", default="GRPANALITICA", help="Target resource group")
    parser.add_argument("--registry-name", default="mlopstestacr", help="Azure Container Registry name (lowercase)")
    parser.add_argument("--image-name", default="mlopstest-api", help="Repository name inside ACR")
    parser.add_argument("--image-version", default="v1", help="Image tag/version")
    parser.add_argument("--dockerfile", default="Dockerfile", help="Path to the Dockerfile")
    args = parser.parse_args()

    root = pathlib.Path(__file__).resolve().parents[1]
    image_tag = f"{args.image_name}:{args.image_version}"
    full_tag = f"{args.registry_name}.azurecr.io/{image_tag}"

    print("Setting Azure subscription...")
    run(["az", "account", "set", "--subscription", args.subscription])

    print("Logging in to ACR...")
    run(["az", "acr", "login", "--name", args.registry_name])

    print("Building Docker image...")
    run([
        "docker",
        "build",
        "-f",
        str(root / args.dockerfile),
        "-t",
        image_tag,
        str(root),
    ])

    print("Tagging image for ACR...")
    run(["docker", "tag", image_tag, full_tag])

    print("Pushing image to ACR...")
    run(["docker", "push", full_tag])

    print(f"Image pushed: {full_tag}")


if __name__ == "__main__":
    main()
