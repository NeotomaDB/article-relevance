CREATE DATABASE neotomarelevance;

CREATE EXTENSION vector;

CREATE DOMAIN doi AS TEXT CHECK (VALUE ~* '^10.\d{4,9}/[-._;()/:A-Z0-9]+$');
COMMENT ON DOMAIN doi IS 'match DOIs (from shoulder)';

CREATE DOMAIN url AS text
CHECK (VALUE ~ 'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{2,255}\.[a-z]{2,9}\y([-a-zA-Z0-9@:%_\+.,~#?!&>//=]*)$');
COMMENT ON DOMAIN url IS 'match URLs (http or https)';

CREATE TABLE papers (
    doi doi,
    title text,
    subtitle text,
    author text,
    subject text,
    abstract text,
    containertitle text,
    language text,
    published text,
    publisher text,
    articleurl url,
    crossrefmeta jsonb,
    dateadded date,
    PRIMARY KEY (doi)
);

CREATE TABLE embeddingmodels (
    embeddingmodelid SERIAL PRIMARY KEY,
    embeddingmodel text,
    dateadded date,
    UNIQUE(embeddingmodel)
);

create table embeddings (
    doi doi REFERENCES papers(doi),
    embeddingmodelid INT REFERENCES embeddingmodels(embeddingmodelid),
    embeddings public.vector,
    embeddingdate date,
    UNIQUE(doi, embeddingmodelid)
);

create table labels (
    labelid SERIAL primary key,
    label text CONSTRAINT no_null NOT NULL,
    labeladded date,
    UNIQUE(label)
);

create table paperlabels (
    doi doi REFERENCES papers(doi),
    labelid INT REFERENCES labels(labelid),
    person text,
    date timestamp,
    UNIQUE(doi, labelid, person)
);

create table models (
    modelid SERIAL PRIMARY KEY,
    modeltype text,
    modelparams jsonb,
    UNIQUE(modeltype, modelparams)
);

create table predictions (
    doi doi REFERENCES papers(doi),
    model INT REFERENCES models(modelid),
    prediction int REFERENCES labels(labelid),
    date timestamp,
    UNIQUE (doi, model, prediction)
);
