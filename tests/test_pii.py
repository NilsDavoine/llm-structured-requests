import pytest

from structreq.catalog import load_catalog
from structreq.pii import find_pii, redact
from structreq.prompt import system_prompt


@pytest.mark.parametrize("text,secret", [
    ("call +352 621 123 456", "621 123 456"),
    ("tel 00352 26 12 34 56", "26 12 34 56"),
    ("portable 691234567", "691234567"),
    ("mobile 691-234-567", "691-234-567"),
    ("UK 0044 20 7946 0123", "7946 0123"),
    ("matricule 1990031512345", "1990031512345"),
    ("mail jean.martin@example.lu", "jean.martin@example.lu"),
    ("IBAN LU28 0019 4006 4475 0000", "LU28 0019 4006 4475 0000"),
    ("pour M. Jean Martin", "Martin"),
    ("de Mme Dupont", "Dupont"),
    ("für Herrn Weber", "Weber"),
    ("Frau Dr. Müller-Lang", "Müller-Lang"),
    ("Hiring Emma Wilson", "Wilson"),
    ("voir avec Sophie", "Sophie"),
    ("habite 12 rue des Lilas", "rue des Lilas"),
    ("lives at 4 Baker Street", "Baker Street"),
    ("wohnt Hauptstraße 5", "Hauptstraße 5"),
    ("code L-1234", "L-1234"),
])
def test_personal_data_is_redacted(text, secret):
    clean, found = redact(text)
    assert found and secret not in clean


@pytest.mark.parametrize("text", [
    "Comptes annuels 2025 au 31.12.2025",
    "relevés 2024 2025, dans 15 jours",
    "TVA du T3, Q4 2025, 3e trimestre",
    "Gründung einer SARL-S, bitte die üblichen Unterlagen.",
    "Annual accounts: bank statements, sales and purchase invoices, within one month.",
    "Constitution d'une SARL-S, il me faut les pièces habituelles.",
])
def test_ordinary_business_text_is_untouched(text):
    assert find_pii(text) == []


def test_catalog_and_instructions_are_clean():
    cat = load_catalog()
    for labels in cat.documents.values():
        for label in labels.values():
            assert find_pii(label) == [], label
    assert find_pii(system_prompt()) == []
