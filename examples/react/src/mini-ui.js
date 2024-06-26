import {Children} from 'react';
import {jsx     } from './jsx.js';
export function Divider() {
  return jsx('hr', null);
}
export function CssBaseline(props) {
  return jsx('div', null, props.children);
}
export function ThemeProvider(props) {
  return jsx('div', null, props.children);
}
export function Container(props) {
  return jsx('div', null, props.children);
}
export function createTheme(settings) {
  return {not: 'implemented'};
}
export function Button(props) {
  const {onClick, disabled} = props;
  return jsx('button', {onClick, disabled}, props.children);
}
export function Grid(props) {
  return jsx('div', null, props.children);
}
export function Alert(props) {
  return jsx('div', null, props.children);
}
export function Stack(props) {
  const {children} = props;
  const trs = Children.map(children, (child, key) => jsx('tr', {key}, jsx('td', {key}, child)));
  return jsx('table', null, jsx('tbody', null, trs));
}
export function Box(props) {
  return jsx('div', null, props.children);
}
export function InputLabel(props) {
  return jsx('div', null, props.children);
}
export function Checkbox(props) {
  return jsx('input', {type: 'checkbox', ...props});
}
export function TextField(TextFieldProps) {
  const {label, disabled, onChange, value, type} = TextFieldProps;
  return (
    jsx('label', null, 
      label,
      jsx('input', {type, disabled, onChange, value}, null)
    )
  )
}
export function Select(SelectProps) {
  const { value, onChange, children } = SelectProps;
  return jsx('select', {value, onChange}, children);
}
export function MenuItem(MenuItemProps) {
  const { value, disabled, children } = MenuItemProps;
  return jsx('option', {value, disabled}, children);
}
export function FormControlLabel(props) {
  return jsx('label', {}, props.label, ' ', props.control);
}
export function FormControl(props) {
  return jsx('div', null, props.children);
}
