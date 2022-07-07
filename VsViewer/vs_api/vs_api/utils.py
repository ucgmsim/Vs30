from functools import wraps
from typing import Iterable, Tuple, Dict, Union, Optional, Type, List

import flask


class MissingKeyError(Exception):
    def __init__(self, key):
        self.error_code = 400
        self.error_msg = f"Request is missing parameter: {key}"


def endpoint_exception_handling(app):
    def endpoint_exception_handling_decorator(f):
        """Handling exception for endpoints"""

        @wraps(f)
        def decorated(*args, **kwargs):
            try:
                return f(*args, **kwargs)

            except MissingKeyError as ex:
                app.logger.error(ex.error_msg, exc_info=True)
                return flask.jsonify({"error": ex.error_msg}), ex.error_code
            except ValueError as ve:
                error_msg = str(ve)
                app.logger.error(error_msg, exc_info=True)
                return (
                    flask.jsonify({"error": error_msg}),
                    400,
                )
            except FileNotFoundError as ex:
                error_msg = f"Result file {ex.filename} does not exist."
                error_code = 404
                return flask.jsonify({"error": error_msg}), error_code
            except Exception as e:
                error_msg = str(e)
                error_code = 500
                return flask.jsonify({"error": error_msg}), error_code

        return decorated

    return endpoint_exception_handling_decorator


def get_check_keys(
    data_dict: Dict,
    keys: Iterable[Union[str, Tuple[str, Type], Tuple[str, Type, any]]],
    optional_keys: Optional[
        Iterable[Union[str, Tuple[str, Type], Tuple[str, Type, any]]]
    ] = None,
) -> Tuple[List[str], Dict[str, object]]:
    """Retrieves the specified keys from the data dict, throws a
    MissingKey exception if one of the keys does not have a value.

    If a type is specified with a key (as a tuple of [key, type]) then the
    value is also converted to the specified type

    If a default is specified with a key and type (as a tuple of [key, type, default]) then the
    value is also converted to the given type and if not specified then that default value is used
    """
    values = []
    for key_val in keys:
        # Check if a type is specified with the key
        if isinstance(key_val, tuple):
            cur_key, cur_type = key_val
        else:
            cur_key, cur_type = key_val, None

        value = data_dict.get(cur_key)
        if value is None:
            raise MissingKeyError(cur_key)

        # Perform a type conversion if one was given & append value
        values.append(value if cur_type is None else cur_type(value))

    optional_values_dict = {}
    if optional_keys is not None:
        for key_val in optional_keys:
            # Check if a type is specified with the key
            if isinstance(key_val, tuple):
                cur_key, cur_type, cur_default = (
                    key_val if len(key_val) == 3 else (*key_val, None)
                )
            else:
                cur_key, cur_type, cur_default = key_val, None, None

            value = data_dict.get(cur_key, cur_default)

            # Perform a type conversion if one was specified
            if value is not cur_default and cur_type is not None:
                value = cur_type(value)

            optional_values_dict[cur_key] = value

    return values, optional_values_dict
