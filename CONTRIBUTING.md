
# Contributing to ThermOHL

Thank you for your interest in ThermOHL.
If you consider contributing to this repository, please read the relevant sections below.

## Questions and issues

The best place to ask questions and report issues is the [Issue](https://github.com/rte-france/pallas-thermohl/issues/) section of the [Github repository](https://github.com/rte-france/pallas-thermohl).
You can use the search function of Github to look for previous similar questions.
For sensitive issues or to request private paid support, you can send a message to <opensource@mews-labs.com>.

## Contributing to the code

Before starting to edit the code to make a change, it is often advisable to open a discussion thread as mentionned above to discuss with the maintainers how it fits in the scope and aims of the project.

When contributing to this project, you must agree that you have authored 100% of the content, that you have the necessary rights to the content and that the content you contribute may be provided under the project licence.

You will then need to submit a pull request (PR). To do so, you will need to:
- Create a fork of ThermOHL on Github (that is a copy on which you have full write access).
- Clone your fork locally on your computer (or add your fork as a new remote to a local git repository).
- On your local git repository, create a new branch dedicated to the bug fix or feature that you want to contribute.
- Create as many commits as you'd like in this branch.
- Push the branch to your fork on Github.
- On Github's web interface, create a new pull request in the main ThermOHL repository at https://github.com/rte-france/pallas-thermohl, requesting to merge the branch on your fork into the main branch of the original ThermOHL repository.
- Until the pull request is accepted, the maintainers can edit the branch on your fork with you.
- Once the pull request is merged, the branch can be deleted or forgotten.

For simple modifications (e.g. typos) most of the process above can be done automatically by Github by using its edition functionality (the pencil icon on the top left of Github's file viewer).

The Github repository includes tests in the `tests` directory. Tests can be run with the pytest package (`python -m pytest`). Before submitting a change of the code, make sure that all tests are passing. If you'd like to add a feature, please add the relevant tests to the pytest directory.

The code follows the PEP8 guidelines for code style. Indentation is done with four spaces. Try to avoid trailing whitespaces whenever possible.

