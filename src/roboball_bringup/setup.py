from glob import glob

from setuptools import find_packages, setup

package_name = 'roboball_bringup'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Roboball team',
    maintainer_email='indraja_babu@berkeley.edu',
    description='Top-level launch, calibration TF, and arm bring-up utilities.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'static_camera_tf = roboball_bringup.static_camera_tf:main',
            'go_home = roboball_bringup.go_home:main',
        ],
    },
)
