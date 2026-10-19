# Terms of use for machine clients

**Service:** the statistics API and the MCP server of the National Statistics
Office (example), stats.example
**Version:** 1.0, 2025-12-01

## 1. What these terms cover

These terms apply to software that retrieves data from the office's API or
MCP server, including AI assistants and agents acting for a user. The data
themselves are licensed under the licence stated in each record; these terms
cover the use of the service.

## 2. Registration and identification

- Clients identify themselves with a client identifier in every request.
- Anonymous access is limited to the public tier and its rate limit.
- A registered client receives a higher rate limit and a contact for notices
  of change.

## 3. Attribution and citation

- A client that presents a value to a person presents the series
  identifier, the reference period, the release date, and the source URL
  returned with the value.
- A client that generates text from the data attributes it to the office.

## 4. Fidelity

- A client does not alter a value, a unit, or a reference period.
- A client marks provisional values as provisional, as the observation
  status field states.
- A client that cannot retrieve a value says so and does not substitute a
  value from another source under the office's name.

## 5. Use of the data for AI

- Training or fine-tuning a model on data retrieved from the service is
  governed by the licence of each product, as stated in the licence field of
  the record and in the licence matrix published with the catalog.
- Where a licence allows training, attribution to the office is kept in the
  model's documentation.

## 6. Limits and changes

- Rate limits are published with the API documentation.
- Changes to the service are announced through the deprecation policy, with
  at least six months' notice before an endpoint is withdrawn.

## 7. Contact

Dissemination unit, data@stats.example.
