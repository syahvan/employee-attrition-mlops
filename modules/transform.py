"""
Author: Syahvan Alviansyah Diva Ritonga
Date: 03/07/2024
This is the transform.py module.
Usage:
- Transform feature and label data
"""

import tensorflow as tf
import tensorflow_transform as tft

CATEGORICAL_FEATURES = {
    'BusinessTravel': 3,
    'Department': 3,
    'Education': 5,
    'EducationField': 6,
    'EnvironmentSatisfaction': 4,
    'Gender': 2,
    'JobInvolvement': 4,
    'JobRole': 9,
    'JobSatisfaction': 4,
    'MaritalStatus': 3,
    'Over18': 2,
    'OverTime': 2,
    'PerformanceRating': 4,
    'RelationshipSatisfaction': 4,
    'WorkLifeBalance': 4
}

NUMERICAL_FEATURES = [
    'Age',
    'DailyRate',
    'DistanceFromHome',
    'EmployeeCount',
    'HourlyRate',
    'JobLevel',
    'MonthlyIncome',
    'MonthlyRate',
    'NumCompaniesWorked',
    'PercentSalaryHike',
    'StandardHours',
    'StockOptionLevel',
    'TotalWorkingYears',
    'TrainingTimesLastYear',
    'YearsAtCompany',
    'YearsInCurrentRole',
    'YearsSinceLastPromotion',
    'YearsWithCurrManager'
]

LABEL_KEY = 'Attrition'


def transformed_name(key):
    """Transform feature key

    Args:
        key (str): the key to be transformed

    Returns:
        str: transformed key
    """

    return f"{key}_xf"


def convert_num_to_one_hot(label_tensor, num_labels=2):
    """Convert a label (0 or 1) into a one-hot vector

    Args:
        label_tensor (int): label tensor (0 or 1)
        num_labels (int, optional): num of label. Defaults to 2.

    Returns:
        tf.Tensor: label tensor
    """

    one_hot_tensor = tf.one_hot(label_tensor, num_labels)
    return tf.reshape(one_hot_tensor, [-1, num_labels])


def replace_nan(tensor):
    """Replace nan value with zero number

    Args:
        tensor (list): list data with na data that want to replace

    Returns:
        list with replaced nan value
    """
    tensor = tf.cast(tensor, tf.float64)
    return tf.where(
        tf.math.is_nan(tensor),
        tft.mean(tensor),
        tensor
    )


def preprocessing_fn(inputs):
    """Preprocess input features into transformed features

    Args:
        inputs (dict): map from feature keys to raw features

    Returns:
        dict: map from features keys to transformed features
    """

    outputs = {}

    for keys, values in CATEGORICAL_FEATURES.items():
        int_value = tft.compute_and_apply_vocabulary(
            inputs[keys], top_k=values+1)
        outputs[transformed_name(keys)] = convert_num_to_one_hot(
            int_value, num_labels=values+1)

    for feature in NUMERICAL_FEATURES:
        inputs[feature] = replace_nan(inputs[feature])
        outputs[transformed_name(feature)] = tft.scale_to_0_1(inputs[feature])

    outputs[transformed_name(LABEL_KEY)] = tf.cast(inputs[LABEL_KEY], tf.int64)

    return outputs
