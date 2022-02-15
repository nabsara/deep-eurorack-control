import adcControl

spifd = adcControl.adc_open()
print(spifd)
if spifd < 0:
    exit(-1)
res = adcControl.adc_bang(spifd)
print(res)
statusVal = adcControl.adc_close(spifd)
print(statusVal)
