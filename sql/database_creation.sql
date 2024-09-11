CREATE DATABASE neotomarelevance;

CREATE EXTENSION vector;

CREATE DOMAIN doi AS TEXT CHECK (VALUE ~* '^10.\d{4,9}/[-._;()/:A-Z0-9]+$');
COMMENT ON DOMAIN doi IS 'match DOIs (from shoulder)';


CREATE DOMAIN url AS text
CHECK (VALUE ~ 'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{2,255}\.[a-z]{2,9}\y([-a-zA-Z0-9@:%_\+.,~#?!&>//=/)/()]*)$');
COMMENT ON DOMAIN url IS 'match URLs (http or https)';

CREATE TABLE IF NOT EXISTS papers (
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
    dateadded timestamp,
    PRIMARY KEY (doi)
);

CREATE TABLE IF NOT EXISTS embeddingmodels (
    embeddingmodelid SERIAL PRIMARY KEY,
    embeddingmodel text,
    UNIQUE(embeddingmodel)
);

create table IF NOT EXISTS embeddings (
    doi doi REFERENCES papers(doi),
    embeddingmodelid INT REFERENCES embeddingmodels(embeddingmodelid),
    embeddings vector(2000),
    date timestamp,
    UNIQUE(doi, embeddingmodelid, embeddings)
);

create table IF NOT EXISTS labels (
    labelid SERIAL primary key,
    label text CONSTRAINT no_null NOT NULL,
    UNIQUE(label)
);

create table IF NOT EXISTS paperlabels (
    doi doi REFERENCES papers(doi),
    labelid INT REFERENCES labels(labelid),
    person text,
    date timestamp,
    UNIQUE(doi, labelid, person)
);

create table IF NOT EXISTS models (
    modelid SERIAL PRIMARY KEY,
    modeltype text,
    modelparams jsonb,
    UNIQUE(modeltype, modelparams)
);

create table IF NOT EXISTS predictions (
    doi doi REFERENCES papers(doi),
    model INT REFERENCES models(modelid),
    prediction int REFERENCES labels(labelid),
    date timestamp,
    UNIQUE (doi, model, prediction)
);

