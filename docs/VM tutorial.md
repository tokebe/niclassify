# Setting up an Ubuntu VM for NIClassify

## Part 1: Creating a Virtual Machine

**Install Virtualbox:** https://www.virtualbox.org/wiki/Downloads

**Download Ubuntu Desktop LTS** (First option on page): https://ubuntu.com/download/desktop



**Launch Virtualbox.** In the main window that opens, **click New** to set up a new machine. Name it however you would like. Ensure that **Type is Linux** and **Version is Ubuntu (64-bit)**. You may also choose to save the machine folder wherever you would like. **Click Continue.**

Next, you have to set your **memory size**. I would recommend a **minimum of 4GB**, however if you have more than 8GB of memory I would recommend setting it higher, to about half of your system memory. The more memory you give to the virtual machine, generally the better it will perform. **Always leave at least 4GB to your system for stable performance.**

On the next screen, leave the selected option as it is. This should be **Create a virtual hard disk now.** The next few screens will walk you through creating a virtual disk.** Click Create.** On the next two screens, you can leave selected default. **Click Continue.** On the "File location and size" screen, choose a size for your virtual disk. I recommend **between 16 and 32 GB.** This will give the machine plenty of room to work with. **Click Create.**

This will create the virtual machine, and it will appear on the list to the left in the main VirtualBox window. We're not quite done yet! There are a number of things we need to configure before we're ready to start running Ubuntu. **Select your new virtual machine and click Settings.**

Now, we have quite a few settings to configure, so I'll run down the line.

1. In the General tab, click Advanced. Change Shared Clipboard 'Bidirectional' and Drag'n'Drop to 'Bidirectional'.

2. In the System tab, click Processor. Change the number of processors to the edge of the green section.

3. In the Display tab, set Video Memory to 128MB, and set Graphics Controller to 'VMSVGA'.

4. In the Storage tab, click on the Empty disk icon (under Controller: IDE). To the right, under Attributes, next to Optical Drive, click the disk dropdown icon. Select 'choose a disk file' and locate the .iso file you downloaded from the Ubuntu website.

From there, you may click OK to exit configuration. Double-click your virtual machine to start it. You'll get a popup asking to select the virtual optical disk. Choose in the dropdown the .iso you just added to the Optical Drive, if it isn't selected by default. **Click Start.**

## Part 2: Setting up Ubuntu

After a few minutes, the machine will boot into the Ubuntu desktop. From here you should be presented with a window that asks you to either try or install Ubuntu. **Click 'Install Ubuntu'.** There are a few setup options here, but for the most part they should be fairly straightforward. You'll want the normal installation, and when prompted ou should **Choose 'Erase disk and install Ubuntu'.** Don't worry, this will only erase the virtual disk. Make sure to use a username and password that are easy to remember.

Once you've finished the setup process, it will take a little while to install Ubuntu. When it finishes, it will prompt you to restart the machine. **Click 'Restart Now'.** 

After a while, you'll have booted back into the Ubuntu desktop, this time with the login you created earlier. Open a terminal and run the following commands:

1. `sudo apt update`

2. `sudo apt upgrade -y`

3. `sudo apt install -y virtualbox-guest-x11`

4. `sudo reboot`

Your machine should reboot again. Once it comes back the the desktop, open a terminal again and run the following command:

`sudo apt install -y git htop`

From here, you should be set up to install NIClassify.

## Part 3: Installing NIClassify

First, we need to clone it:

`git clone https://github.com/tokebe/niclassify.git`

We want to cd into the newly-created directory:

`cd niclassify`

Then we want to make sure we've set it to the latest release, as opposed to the latest code:

`git describe --tags --abbrev=0`

This should give you the latest release tag. At the time of writing, that should be `v1.0.8-beta`. Knowing this tag, we can now change to that version:

`git checkout v1.0.8-beta` (replacing `v1.0.8-beta` with whatever the latest version tag is, which should be the output from the previous command)

From here, we want to properly install NIClassify:

`./launch-linux.sh`

You will be prompted for your password, type it in and hit enter. From here, a lot of installation stuff will start happening, and it could take a while - even more than an hour, thanks to how long package installation for R takes on linux. Grab a cup of coffee and do something else until it's done. When it is done, you should the the NIClassify window sitting there waiting for you! 

## Launching NIClassify

In the future, launching NIClassify should be as simple as navigating to the launcher in the file explorer and running it, or, from the terminal:

`cd niclassify && ./launch-linux.sh

## Updating NIClassify

When new updates come out, updating is just a few steps. Open a new terminal and change directory to niclassify's folder:

`cd niclassify`

We need to leave the release tag and then we can pull the latest changes:

`git switch - && git pull`

Then it's just a matter of finding the latest tag:

`git describe --tag --abbrev=0`

And then switching to that tag:

`git checkout v1.0.8-beta` (replace `v1.0.8-beta` with the output of the previous command)


