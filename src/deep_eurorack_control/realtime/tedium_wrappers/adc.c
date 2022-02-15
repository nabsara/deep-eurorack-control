/***********************************************************************
* This header file contains the mcp3208 Spi class definition.
* Its main purpose is to communicate with the MCP3208 chip using
* the userspace spidev facility.
* The class contains four variables:
* mode        -> defines the SPI mode used. In our case it is SPI_MODE_0.
* bitsPerWord -> defines the bit width of the data transmitted.
*        This is normally 8. Experimentation with other values
*        didn't work for me
* speed       -> Bus speed or SPI clock frequency. According to
*                https://projects.drogon.net/understanding-spi-on-the-raspberry-pi/
*            It can be only 0.5, 1, 2, 4, 8, 16, 32 MHz.
*                Will use 1MHz for now and test it further.
* spifd       -> file descriptor for the SPI device
*
* edit mxmxmx: adapted for mcp3208 / terminal tedium
*
* CVs are setup this way on the panel:
* 1, 2, 3
* 4, 5, 6
*
* ****************************************************************************/
#include <unistd.h>
#include <stdint.h>
#include <stdio.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <fcntl.h>
#include <linux/spi/spidev.h>

// Redefine pd_error() to just log everything here
void pd_error(int spifd, const char *fmt, ...) {
    va_list ap;
    printf(fmt, ap);
}

/***********************************************************
* terminal_tedium_adc_close(): Responsible for closing the spidev interface.
* *********************************************************/
int terminal_tedium_adc_close(int spifd)
{
    int statusVal = -1;
    if (spifd == -1) {
        pd_error(spifd, "terminal_tedium_adc: device not open\n");
        return -1;
    }
    statusVal = close(spifd);
    if (statusVal < 0) {
        pd_error(spifd, "terminal_tedium_adc: could not close SPI device\n");
        exit(1);
    }
    return statusVal;
}


/**********************************************************
* terminal_tedium_adc_open() :function is called by the "open" command
* It is responsible for opening the spidev device
* and then setting up the spidev interface.
* member variables are used to configure spidev.
* They must be set appropriately before calling
* this function.
* *********************************************************/
int terminal_tedium_adc_open()
{
    unsigned char mode = SPI_MODE_0;
    unsigned char bitsPerWord = 8;
    unsigned int speed = 4000000;

    int statusVal = 0;  // Error code value
    int spifd = -1;
    // Open fd
    spifd = open("/dev/spidev0.1", O_RDWR);  // we're using CS1
    if (spifd < 0) {
        statusVal = -1;
        pd_error(spifd, "could not open SPI device\n");
        return -1;
    }
    // Setup TX/RX params
    statusVal = ioctl(spifd, SPI_IOC_WR_MODE, &(mode));
    if (statusVal < 0){
        pd_error(spifd, "Could not set SPIMode (WR)...ioctl fail\n");
        terminal_tedium_adc_close(spifd);
        return -1;
    }
    statusVal = ioctl(spifd, SPI_IOC_RD_MODE, &(mode));
    if (statusVal < 0) {
        pd_error(spifd, "Could not set SPIMode (RD)...ioctl fail\n");
        terminal_tedium_adc_close(spifd);
        return -1;
    }
    statusVal = ioctl(spifd, SPI_IOC_WR_BITS_PER_WORD, &(bitsPerWord));
    if (statusVal < 0) {
        pd_error(spifd, "Could not set SPI bitsPerWord (WR)...ioctl fail\n");
        terminal_tedium_adc_close(spifd);
        return -1;
    }
    statusVal = ioctl(spifd, SPI_IOC_RD_BITS_PER_WORD, &(bitsPerWord));
    if (statusVal < 0) {
        pd_error(spifd, "Could not set SPI bitsPerWord(RD)...ioctl fail\n");
        terminal_tedium_adc_close(spifd);
        return -1;
    }
    statusVal = ioctl(spifd, SPI_IOC_WR_MAX_SPEED_HZ, &(speed));
    if (statusVal < 0) {
        pd_error(spifd, "Could not set SPI speed (WR)...ioctl fail\n");
        terminal_tedium_adc_close(spifd);
        return -1;
    }
    statusVal = ioctl(spifd, SPI_IOC_RD_MAX_SPEED_HZ, &(speed));
    if (statusVal < 0) {
        pd_error(spifd, "Could not set SPI speed (RD)...ioctl fail\n");
        terminal_tedium_adc_close(spifd);
        return -1;
    }
    return spifd;
}


/********************************************************************
* This function writes data "data" of length "length" to the spidev
* device. Data shifted in from the spidev device is saved back into
* "data".
* ******************************************************************/
int terminal_tedium_adc_write_read(int spifd, unsigned char *data, int length)
{
    struct spi_ioc_transfer spid[length];
    int i = 0;
    int retVal = -1;

    // one spi transfer for each byte
    for (i = 0 ; i < length ; i++) {
        memset (&spid[i], 0x0, sizeof (spid[i]));
        spid[i].tx_buf        = (unsigned long)(data + i); // transmit from "data"
        spid[i].rx_buf        = (unsigned long)(data + i); // receive into "data"
        spid[i].len           = sizeof(*(data + i));
        spid[i].speed_hz      = 4000000;
        spid[i].bits_per_word = 8;
    }
    // ioctl() triggers the actual reading
    retVal = ioctl(spifd, SPI_IOC_MESSAGE(length), &spid);
    if(retVal < 0)  // on success zero is returned
        pd_error(spifd, "problem transmitting spi dataioctl\n");
    return retVal;
}

/***********************************************************************
* mcp3208 enabled external that by default interacts with /dev/spidev0.0 device using
* terminal_tedium_adc_MODE_0 (MODE 0) (defined in linux/spi/spidev.h), speed = 1MHz &
* bitsPerWord=8.
* *********************************************************************/
int terminal_tedium_adc_bang(int *a2d, int spifd)
{
    int a2dVal[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    int a2dChannel = 0;
    unsigned char data[3];
    int numChannels = 0x6; // 6 channels for wm8731 version
    int SCALE = 4001; // make nice zeros ...

    if (spifd == -1) {
        pd_error(spifd, "device not open\n");
        return -1;
    }

    unsigned int SMOOTH = 1;
    unsigned int SMOOTH_SHIFT = 0;
    int DEADBAND = 1;
    // Prepare the
    for (unsigned int i = 0; i < SMOOTH; i++) {
        for (a2dChannel = 0; a2dChannel < numChannels; a2dChannel++) {
            data[0]  =  0x06 | ((a2dChannel >> 2) & 0x01);
            data[1]  =  a2dChannel << 6;
            data[2]  =  0x00;
            if (terminal_tedium_adc_write_read(spifd, data, 3) < 0)
                return -1;
            a2dVal[a2dChannel] += (((data[1] & 0x0f) << 0x08) | data[2]);
        }
    }

    for (a2dChannel = 0; a2dChannel < numChannels; a2dChannel++) {
        if (DEADBAND) {
            int tmp  = SCALE - (a2dVal[a2dChannel] >> SMOOTH_SHIFT);
            int tmp2 = a2d[a2dChannel];

            if ((tmp2 - tmp) > DEADBAND || (tmp - tmp2) > DEADBAND) {
                a2dVal[a2dChannel] = tmp < 0 ? 0 : tmp;
            } else {
                a2dVal[a2dChannel] = tmp2;
            }
            a2d[a2dChannel] = a2dVal[a2dChannel];
        } else {
            int tmp = SCALE - (a2dVal[a2dChannel] >> SMOOTH_SHIFT);
            a2dVal[a2dChannel] = tmp < 0 ? 0 : tmp;
        }
    }
    return 1;
}


int main()
{
    int statusVal = -1;
    int a2d[8];

    int spifd = terminal_tedium_adc_open();
    printf("spifd %d\n", spifd);
    if (spifd < 0) {
        exit(-1);
    }
    statusVal = terminal_tedium_adc_bang(a2d, spifd);
    if (statusVal < 0) {
        exit(-1);
    }
    printf("3- read done\n");
    for (int i = 0 ; i < 6 ; i++)
        printf("value CV%d: %d\n", i, a2d[i]);
    terminal_tedium_adc_close(spifd);
    return statusVal;
}
