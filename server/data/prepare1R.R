
roundmax = function(val, maxdp=6) {
    # round to decimal places, don't add trailing zeros
    return(round((val + .Machine$double.eps) * 10^maxdp) / 10^maxdp)
}
