from datetime import datetime
from .check_apis import paper_label_exists, project_exists, label_exists, person_exists
from .register_apis import register_paper_label, register_person, register_label

def add_paper_labels(labellist: list,
                     project: str,
                     create: bool = False):
    """_Add label data for DOIs in the set of metadata. Allows the user to pass in a `source`._

    Args:
        labellist (list): _A list of dict objects containing the keys `label`, `person` and `doi`._
        project (str): _A valid project name in the database, registered with the `register_project()` function.
        create (bool, optional): _If no label data exists, should it be created in the cloud?_. Defaults to False.

    Returns:
        _dict_: _A list of dict objects returned from `register_paper_label()`._
    """
    valid_project = project_exists(project)
    registry = []
    # Check that the project exists:
    if valid_project is None:
        raise ValueError(f'The project {project} is not registered in the database. Use `register_project()` to add the project before adding labels.')
    # Check that each of the labels are valid:
    for i in set([j.get('label') for j in labellist]):
        valid_label = label_exists(i, project)
        if valid_label is None and create is False:
            raise ValueError(f"The label {i} doesn't exist for project {project}. To add this label set `create` to True.")
        elif valid_label is None and create is True:
            new_label = register_label(i, project)
    # Check that each of the people are valid:
    for i in set([j.get('person') for j in labellist]):
        valid_person = person_exists(i)
        if valid_label is None and create is False:
            raise ValueError(f"The label {i} doesn't exist for project {project}. To add this label set `create` to True.")
        elif valid_label is None and create is True:
            new_person = register_person(i)
    # Label papers:
    for i in labellist:
        paper_in = paper_label_exists(doi = i.get('doi'), label = i.get('label'), project = project, person = i.get('person'))
        if paper_in is None:
            labeledpaper = register_paper_label(i.get('doi'), i.get('label'), project, i.get('person'))
            if labeledpaper is not None:
                break
            registry.append(labeledpaper)
    return registry
