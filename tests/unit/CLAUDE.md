# Unit Tests

## Purpose

Isolated tests for individual modules. Each test file mirrors a source module in `nba_model/`.

## Structure

| File | Tests | Source Module |
|------|-------|---------------|
| `test_cli.py` | CLI commands work | `nba_model/cli.py` |
| `test_config.py` | Settings validation | `nba_model/config.py` |

## Subdirectories

Mirror the `nba_model/` subpackage structure:
- `data/` → tests for `nba_model/data/`
- `features/` → tests for `nba_model/features/`
- `models/` → tests for `nba_model/models/`
- etc.

## Test Patterns

```python
class TestClassName:
    """Tests for ClassName."""
    
    def test_method_does_something(self) -> None:
        """Method should do something when given input."""
        # Arrange
        input_data = ...
        
        # Act
        result = method(input_data)
        
        # Assert
        assert result == expected

    def test_method_raises_on_invalid(self) -> None:
        """Method should raise ValueError on invalid input."""
        with pytest.raises(ValueError, match="specific message"):
            method(invalid_input)
```

## Fixtures (in conftest.py)

- `tmp_path` - Temporary directory (pytest builtin)
- `monkeypatch` - Environment variable mocking
- `sample_game` - Sample Game object (to be added)
- `sample_features` - Sample feature array (to be added)

## Anti-Patterns

- ❌ Never import from other test files
- ❌ Never test multiple behaviors in one test
- ❌ Never use `assert True` or `assert False` alone
- ❌ Never skip type hints in test signatures
