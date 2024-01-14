# --------------------------------------- HEADER -------------------------------------------
# -*- coding: utf-8 -*-
# Copyright (c) 2024 Roman Usoltsev
# https://www.linkedin.com/in/algoviator/
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ---------------------------------- IMPORT LIBRARIES --------------------------------------

import yaml

# ---------------------------------- CLASSES & FUNCTIONS -----------------------------------


class LoadConfig(object):
    """
    A class for loading configuration data from a YAML file.

    This class provides methods to determine the configuration filename based on the calling module's name
    and load the configuration data from the specified file path.

    Attributes:
        filename (str): The configuration filename.
        data (dict or None): The loaded configuration data or None if the file is not found.
    """
    def __init__(self, module_name=''):

        # Initialize the class by loading the configuration data
        self.data = self.load_config(file_path=f'{module_name}')

    @staticmethod
    def load_config(file_path=''):
        """
        Load configuration data from the specified file path.

        Args:
            file_path (str): The path to the configuration file.

        Returns:
            dict or None: The loaded configuration data or None if the file is not found.
        """
        try:
            with open(file_path, 'r') as file:
                config = yaml.safe_load(file)
            return config

        except FileNotFoundError:
            print(f'Config file \'{file_path}\' not found.')
            return None
