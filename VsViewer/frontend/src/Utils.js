export const roundValue = (value, dataIsKPa = false) => {
  if (dataIsKPa) {
    let tempValue = parseFloat(value) / 1000;
    return parseFloat(tempValue.toPrecision(4));
  } else {
    return parseFloat(parseFloat(value).toPrecision(4));
  }
};

export const errorCheckWeights = (weights) => {
  let sum = 0;
  let result = true;
  try {
    for (const value of Object.values(weights)) {
      let floatRegex = /^-?\d+(?:[.,]\d*?)?$/;
      if (!floatRegex.test(value)) {
        result = false;
      }
      sum += parseFloat(value);
    }
    if (!(sum >= 0.98 && sum <= 1.02)) {
      result = false;
    }
  } catch (e) {
    result = false;
  }
  return result;
};

export const errorCheckFloatInput = (input) => {
  let floatRegex = /^-?\d+(?:[.,]\d*?)?$/;
  return floatRegex.test(input);
};
