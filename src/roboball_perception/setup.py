from setuptools import find_packages, setup

package_name = 'roboball_perception'

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
    description='Ball detection and state estimation for Roboball.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ball_detector = roboball_perception.ball_detector:main',
        ],
    },
)
