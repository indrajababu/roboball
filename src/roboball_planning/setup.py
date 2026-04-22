from setuptools import find_packages, setup

package_name = 'roboball_planning'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Roboball team',
    maintainer_email='indraja_babu@berkeley.edu',
    description='Ballistic prediction, IK, trajectory generation, and arm execution.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ik = roboball_planning.ik:main',
            'trajectory_predictor = roboball_planning.trajectory_predictor:main',
            'strike_planner = roboball_planning.strike_planner:main',
            'validate_trajectory = roboball_planning.validate_trajectory:main',
        ],
    },
)
