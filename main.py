import subprocess
import platform


def get_gpu_count():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
        gpu_count = len(result.stdout.decode('utf-8').strip().split('\n'))
        return gpu_count
    except Exception as e:
        print(f"Error detecting GPUs: {e}")
        return 0


def get_os_type():
    os_type = platform.system()
    return os_type


def generate_docker_compose(gpu_count, os_type):
    image = "miislab/aoi_cuda11.8:windows" if os_type == "Windows" else "miislab/aoi_cuda11.8:ubuntu"
    with open('docker-compose-template.yml', 'r') as file:
        compose_template = file.read()

    compose_content = compose_template.replace('GPU_COUNT_PLACEHOLDER', str(gpu_count)).replace('IMAGE_PLACEHOLDER',
                                                                                                image)

    with open('docker-compose.yml', 'w') as file:
        file.write(compose_content)

    print("Docker Compose file generated successfully.")


def run_docker_compose():
    try:
        subprocess.run(['docker-compose', 'up'], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running docker-compose: {e}")


def main():
    gpu_count = get_gpu_count()
    os_type = get_os_type()
    generate_docker_compose(gpu_count, os_type)
    run_docker_compose()


if __name__ == "__main__":
    main()
