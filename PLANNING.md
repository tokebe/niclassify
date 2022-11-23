# Redo all steps into procedural functions for easy command-line or gui usage

- All steps are to be re-grouped into sensible steps.
- Functions will use callbacks to prompt when necessary. A request object/dictionary will be used to generalize the requirements of a prompt.
- Everything a function needs will be provided by argument, except decision making which may be provided by prompt.
- No tempfiles will be used, no especially fancy anything (no threading, etc -- that is the job of the interface).
- All options will be provided for by argument, and any optional not provided will use a sensible default.
- All functions will log as much useful information as possible, including warnings, result of user choice, etc.
- All functions which split files will have the option to output split files based on name

Step reorganization:

- `match` shall handle looking up sequences against BOLD for species identification
  - will only accept identifications with given certainty (sensible default required)
- `get` shall handle BOLD lookups to get new data
  - option to append to provided output file, doesn't fail if file doesn't exist
- `intake` shall cover everything up through 'filter sequences'
  - multiple input files, handles merging and prompting
- `align` shall handle aligning sequences
- `delimit` shall handle delimiting OTUs
- `featgen` shall handle generating features
- `lookup` shall handle looking up known statuses
- `train` shall train on given data, outputting the classifier and reports
- `predict` shall use the given classifier to predict on the given data

# Develop a TUI for ease-of-use in certain circumstances

- The TUI will use prompt toolkit/etc to make use as easy as possible
- The TUI can be used as a launcher for the GUI
- The TUI will use the command `wizard` to run the user through the steps without needing to know how to use the command-line options (but will display the commands)

# Rebuild the GUI

IDK, either just take the current one and clean it up or write a whole Electron frontend I couldn't say atm
