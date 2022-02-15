/*
 *  left outlet: print time
 *  right outlet: print 1 when pushed, 0 when released
 */
#include <stdio.h>
// #include <wiringPi.h>

t_class *tedium_switch_class;

typedef struct _tedium_switch
{
	t_clock *x_clock;
	t_int clkState;
	t_int switchState;
	t_int ticks;
	t_int pinNum;

} t_tedium_switch;

void tedium_switch_tick(t_tedium_switch *x)
{
	int prevState = x->clkState;
	x->clkState = digitalRead(x->pinNum);
	// pin pulled low since last tick ?
	if(prevState && !x->clkState) {
		x->switchState = 0x1;
		outlet_float(x->x_out2, 0x1);
	}
	// released ?
	if (!prevState && x->clkState) {
		outlet_float(x->x_out1, x->ticks);
		outlet_float(x->x_out2, 0x0);
		x->switchState = 0x0;
		x->ticks = 0x0;
	}
	// delay 1 msec
	clock_delay(x->x_clock, 0x1);
	// if button is held, count++
	if (x->switchState == 0x1) {
		x->ticks++;
	}
}

void *tedium_switch_new(t_floatarg _pin)
{
	t_tedium_switch *x = (t_tedium_switch *)pd_new(tedium_switch_class);
	x->x_clock = clock_new(x, (t_method)tedium_switch_tick);
	// valid pin?
	if (_pin == 23 || _pin == 24 || _pin == 25) x->pinNum = _pin;
	else x->pinNum = 23; // default to pin #23
	// init
	x->clkState = 1;
	x->switchState = 0;
	x->ticks = 0;
	pinMode(x->pinNum, INPUT);
	pullUpDnControl(x->pinNum, PUD_UP);
	tedium_switch_tick(x);
	return (void *)x;
}

int main(int ac, char **ac)
{
    t_tedium_switch *x = tedium_switch_new()
    return 1;
}
