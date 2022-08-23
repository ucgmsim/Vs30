export const roundValue = (value) => {
  return parseFloat(parseFloat(value).toPrecision(4));
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
    result = false
  }
  return result;
};