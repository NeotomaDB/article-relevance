from datetime import datetime
from .check_apis import paper_label_exists, project_exists, label_exists, person_exists
from .register_apis import register_paper_label, register_person, register_label

def add_paper_labels(labellist: list, project: str, create: bool = False):
    """_Add label data for DOIs in the set of metadata. Allows the user to pass in a `source`._

    Args:
        label_store (dict): _A dict with elements `Bucket` and `Key` to identify the file source._
        label_df (pd.DataFrame): _A pd.DataFrame with columns `doi` and `label` (optional `source`)._
        source (str, optional): _The label source, for example "From DB" or "Simon Goring"_. Defaults to None.
        create (bool, optional): _If no label data exists, should it be created in the cloud?_. Defaults to False.

    Returns:
        _DataFrame_: _A Pandas DataFrame with columns `doi`, `label`, `source` and `date`._
    """
    valid_project = project_exists(project)
    registry = []
    if valid_project is None:
        raise ValueError(f'The project {project} is not registered in the database. Use `register_project()` to add the project before adding labels.')
    for i in set([j.get('label') for j in labellist]):
        valid_label = label_exists(i, project)
        if valid_label is None and create is False:
            raise ValueError(f"The label {i} doesn't exist for project {project}. To add this label set `create` to True.")
        elif valid_label is None and create is True:
            new_label = register_label(i, project)
    for i in set([j.get('person') for j in labellist]):
        valid_person = person_exists(i)
        if valid_label is None and create is False:
            raise ValueError(f"The label {i} doesn't exist for project {project}. To add this label set `create` to True.")
        elif valid_label is None and create is True:
            new_person = register_person(i)
    for i in labellist:
        paper_in = paper_label_exists(doi = i.get('doi'), label = i.get('label'), project = project, person = i.get('person'))
        if paper_in is None:
            labeledpaper = register_paper_label(i.get('doi'), i.get('label'), project, i.get('person'))
            if labeledpaper is not None:
                break
            registry.append(labeledpaper)
    return registry
